from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
)

from .llava.model.llava_jack_gpt4roi import LlavaLlamaForCausalLM
# from .segment_anything import build_sam_vit_h
from .GroundingDINO.groundingdino import build_groundingdino
from utils.misc import NestedTensor, nested_tensor_from_tensor_list
from gpt4roi.models.layers import MLVLROIQueryModule


class ChatterBox(nn.Module):
    def __init__(
            self,
            local_rank,
            vg_token_idx,  # [VG] id
            tokenizer,
            llm_version,
            lora_r,
            precision,
            load_in_4bit=False,
            load_in_8bit=False,
            lora_target_modules=["q_proj", "v_proj"],
            lora_alpha=16,
            lora_dropout=0.05,
            vision_tower="openai/clip-vit-large-patch14",
            mm_vision_select_layer=-2,
            freeze_lm=True,
            out_dim=256,
            ce_loss_weight=1.0,
            dice_loss_weight=0.5,
            bce_loss_weight=2.0,
            vision_branch_args=None,
    ):
        super().__init__()
        self.local_rank = local_rank
        self.tokenizer = tokenizer
        self.image_token = tokenizer.cls_token_id  # what is it ?
        self.precision = precision
        self.ce_loss_weight = ce_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.bce_loss_weight = bce_loss_weight

        # LLaVA
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        num_new_tokens = tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, '<bbox>'], special_tokens=True
            # do we need add region start/end tokens
        )

        if precision == "bf16":
            self.lm = LlavaLlamaForCausalLM.from_pretrained(
                llm_version,
                torch_dtype=torch.bfloat16,
                cache_dir=None,
                low_cpu_mem_usage=True,
            )
        elif precision == "fp16":
            if load_in_4bit:
                self.lm = LlavaLlamaForCausalLM.from_pretrained(
                    llm_version,
                    load_in_4bit=True,
                    cache_dir=None,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    ),
                )
            elif load_in_8bit:
                self.lm = LlavaLlamaForCausalLM.from_pretrained(
                    llm_version,
                    load_in_8bit=True,
                    cache_dir=None,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                )
            else:
                self.lm = LlavaLlamaForCausalLM.from_pretrained(
                    llm_version,
                    torch_dtype=torch.half,
                    cache_dir=None,
                    low_cpu_mem_usage=True,
                )
        else:
            self.lm = LlavaLlamaForCausalLM.from_pretrained(
                llm_version,
                torch_dtype=torch.float32,
                cache_dir=None,
                low_cpu_mem_usage=True,
            )

        self.lm.enable_input_require_grads()
        self.lm.gradient_checkpointing_enable()
        self.lm.config.use_cache = False

        model_vision_dict = self.lm.get_model().initialize_vision_modules(
            vision_tower=vision_tower,
            mm_vision_select_layer=mm_vision_select_layer,
            precision=precision,
        )
        vision_config = model_vision_dict["vision_config"]
        vision_tower = self.lm.get_model().vision_tower[0]

        self.lm.model.config.eos_token_id = tokenizer.eos_token_id
        self.lm.model.config.bos_token_id = tokenizer.bos_token_id
        self.lm.model.config.pad_token_id = tokenizer.pad_token_id

        if vision_tower.device.type == "meta":
            if precision == "bf16":
                vision_tower = CLIPVisionModel.from_pretrained(
                    vision_tower.config._name_or_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                ).cuda(local_rank)
            elif precision == "fp16":
                vision_tower = CLIPVisionModel.from_pretrained(
                    vision_tower.config._name_or_path,
                    torch_dtype=torch.half,
                    low_cpu_mem_usage=True,
                ).cuda(local_rank)
            else:
                vision_tower = CLIPVisionModel.from_pretrained(
                    vision_tower.config._name_or_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                ).cuda(local_rank)
            self.lm.get_model().vision_tower[0] = vision_tower
        else:
            if precision == "bf16":
                vision_tower.to(device="cuda", dtype=torch.bfloat16)
            elif precision == "fp16":
                vision_tower.to(device="cuda", dtype=torch.half)
            else:
                vision_tower.to(device="cuda", dtype=torch.float32)

        self.lm.config.tune_mm_mlp_adapter = False
        self.lm.config.freeze_mm_mlp_adapter = False
        self.lm.config.mm_use_im_start_end = True
        vision_config.use_im_start_end = True
        self.lm.config.sep_image_conv_front = False

        self.lm.initialize_vision_tokenizer(
            mm_use_im_start_end=True,
            tokenizer=tokenizer,
            num_new_tokens=num_new_tokens,
            device=local_rank,
            tune_mm_mlp_adapter=False,
        )
        if freeze_lm:
            for n, param in self.lm.named_parameters():
                param.requires_grad = False

        # LoRA
        if lora_r > 0:
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.lm = get_peft_model(self.lm, config)
            self.lm.print_trainable_parameters()

        self.llm_version = llm_version

        self.vg_token_idx = vg_token_idx
        self.lm.resize_token_embeddings(len(tokenizer))

        for n, p in self.lm.named_parameters():
            if any([x in n for x in ["lm_head", "embed_tokens"]]) and p.shape[0] == len(
                    tokenizer
            ):
                p.requires_grad = True

        # DINO module
        self.visual_grounding_model, self.criterion_grounding = build_groundingdino(vision_branch_args)
        if precision == "bf16":
            self.visual_grounding_model.to(device='cuda', dtype=torch.bfloat16)
        elif precision == 'fp16':
            self.visual_grounding_model.to(device='cuda', dtype=torch.half)
        else:
            self.visual_grounding_model.to(device='cuda', dtype=torch.float32)

        ##################################################### learnable #########################################################
        self.spi_module = MLVLROIQueryModule(embed_dims=1024, out_dims=5120, num_levels=4)
        
        for n, param in self.spi_module.named_parameters():
            param.requires_grad = True
        

        self.spi_module.to(device='cuda', dtype=torch.float32)

        self.bbox_token_id = tokenizer.convert_tokens_to_ids(['<bbox>'])[0]

        # Projection layer
        in_dim = self.lm.config.hidden_size
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])

        self.tgt_align = nn.Linear(out_dim, 256)
        self.refpoint_align = nn.Linear(out_dim, 4)

        
    def load_vision_dict(self, state_dict, strict=False):
        msg = self.visual_grounding_model.load_state_dict(state_dict, strict=strict)
        print('loading visual_grounding_model ', msg)

    def get_visual_embs(self, samples):
        with torch.no_grad():
            features, poss = self.visual_grounding_model.forward_backbone(samples)
        return features, poss

    def forward(
            self,
            images: torch.FloatTensor,
            images_clip: torch.FloatTensor,
            regions_lists,
            input_ids: torch.LongTensor,
            labels: torch.LongTensor,
            attention_masks: torch.LongTensor,
            offset: torch.LongTensor=None,
            bboxes_gt_list: List[torch.FloatTensor]=None,
            label_gt_list: List[torch.FloatTensor]=None,
            inference: bool = False,
            image_path=None,
            **kwargs
    ):

        # print('bboxes_gt_list  >>> ', bboxes_gt_list, label_gt_list)
        # print(images.shape)
        batch_size = images.shape[0]
        assert batch_size == len(offset) - 1  # note here !!!!

        vg_token_mask = input_ids[:, 1:] == self.vg_token_idx
        vg_token_mask = torch.cat(
            [
                vg_token_mask,
                torch.zeros((vg_token_mask.shape[0], 1)).bool().cuda(self.local_rank),
            ],
            dim=1,
        )

        #################################################################################
        bboxes_list = []
        if sum([len(bbox) for bbox in regions_lists]) > 0:
            for box in regions_lists:
                bboxes_list.append(box.half().cuda(self.local_rank))


        images_clip_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            images_clip_i = (
                images_clip[i]
                .unsqueeze(0)
                .expand(end_i - start_i, -1, -1, -1)  # why here expand?
                .contiguous()
            )
            images_clip_list.append(images_clip_i)
        images_clip = torch.cat(images_clip_list, dim=0)

        output = self.lm(
            images=images_clip,
            attention_mask=attention_masks,
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
            image_path=image_path,
            bboxes=bboxes_list,
            bbox_token_id=self.bbox_token_id,
            spi_module=self.spi_module
        )

        output_hidden_states = output.hidden_states

        if sum([len(bboxes) for bboxes in bboxes_gt_list]) > 0:  # this batch has bboxes to predict

            if isinstance(images, (list, torch.Tensor)):
                samples = nested_tensor_from_tensor_list(images)
            else:
                samples = images

            dtype = images.dtype
            if dtype == torch.float16:
                with torch.cuda.amp.autocast(enabled=True):
                    features, poss = self.get_visual_embs(samples)
            else:
                features, poss = self.get_visual_embs(samples)

            hidden_states = []

            assert len(self.text_hidden_fcs) == 1
            hidden_states.append(self.text_hidden_fcs[0](output_hidden_states[-1]))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

            pred_embeddings = last_hidden_state[vg_token_mask]
            initial_pred_embeddings = pred_embeddings
            
            vg_token_counts = vg_token_mask.int().sum(-1)  # [bs, ]
            vg_token_offset = vg_token_counts.cumsum(-1)
            vg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), vg_token_offset], dim=0
            )

            vg_token_offset = vg_token_offset[offset]

            pred_embeddings_ = []
            for i in range(len(vg_token_offset) - 1):
                start_i, end_i = vg_token_offset[i], vg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            vg_loss = torch.tensor(0.0).to(output['loss'].device)
            target = []
            gt_ids = []
            for i in range(len(pred_embeddings)):  # process each data
                if len(bboxes_gt_list[i]) == 0:  # for VQA dataset -> no bbox prediction
                    continue
                gt_ids.append(i)
                bboxes = bboxes_gt_list[i]
                labels = label_gt_list[i]      
                target.append({
                        'boxes': bboxes[0].to(images.device),  # tensor([[0.6243, 0.0892, 0.0247, 0.0820], [0.4349, 0.1022, 0.3568, 0.2044]]), 
                        'labels': torch.tensor([0]).to(images.device)
                })
            sample = NestedTensor(
                tensors=samples.tensors[gt_ids],
                mask=samples.mask[gt_ids]
            )
            feature = []
            for feat in features:
                feature.append(
                    NestedTensor(
                        tensors=feat.tensors[gt_ids],
                        mask=feat.mask[gt_ids]
                    )
                )

            pos = []
            for p in poss:
                pos.append(p[gt_ids]) 

            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.visual_grounding_model.forward_enc_dec(sample, feature, pos, query_embedding=initial_pred_embeddings[gt_ids].unsqueeze(1))
                # print(outputs)
                loss_dict = self.criterion_grounding(outputs, target, initial_pred_embeddings[gt_ids].unsqueeze(1))

                weight_dict = self.criterion_grounding.weight_dict

                loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            vg_loss += loss


            output['loss'] *= 0
        else:
            vg_loss = torch.tensor(0.0).to(output['loss'].device)

        return {
            "vqa_loss": output['loss'],
            "vg_loss": vg_loss,
        }
    def evaluate(
            self,
            images_clip,
            images,
            bboxes_list,
            input_ids,
            max_new_tokens=32,
    ):

        with torch.no_grad():
            outputs = self.lm.generate(
                images=images_clip,
                bboxes=bboxes_list,
                bbox_token_id=self.bbox_token_id,
                spi_module=self.spi_module,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states
            output_ids = outputs.sequences
        if isinstance(images, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(images)
        else:
            samples = images


        dtype = images.dtype
        if dtype == torch.float16:
            with torch.cuda.amp.autocast(enabled=True):
                features, poss = self.get_visual_embs(samples)
        else:
            features, poss = self.get_visual_embs(samples)

  
        vg_token_mask = output_ids[:, 1:] == self.vg_token_idx
        vg_token_mask = torch.cat(
            [
                vg_token_mask,
            ],
            dim=1,
        )
        pred_boxes = []
        if vg_token_mask.sum():
            hidden_states = []
            assert len(self.text_hidden_fcs) == 1
            hidden_states.append(self.text_hidden_fcs[0](output_hidden_states[-1]))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

            pred_embeddings = last_hidden_state[vg_token_mask]
            pred_embeddings = pred_embeddings[-1]
            pred_embeddings = pred_embeddings.unsqueeze(0)
            pred_embeddings = [pred_embeddings]
            
            for i in range(len(pred_embeddings)):  # process each data
    
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = self.visual_grounding_model.forward_enc_dec(samples, features, poss, query_embedding=pred_embeddings[i].unsqueeze(1))
                    pred_boxes.append(outputs)
        return output_ids, pred_boxes