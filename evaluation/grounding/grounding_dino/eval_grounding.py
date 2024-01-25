import argparse
import os
import sys
import cv2
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, CLIPImageProcessor
from model.ChatterBox_Referrring_Grounding_grounding_dino import JACK
from utils.conversation import get_default_conv_template
import utils.transforms as T
from PIL import Image
import json
from utils.slconfig import DictAction, SLConfig
import torchvision


def parse_args(args):
    parser = argparse.ArgumentParser(description="JACK chat")
    parser.add_argument("--version", default="/path/to/llava-llama-2-13b-chat-lightning-preview")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument("--vision_pretrained", default="PATH_TO_DINO", type=str)
    parser.add_argument("--weight", default="/path/to/chatterbox_grounding_ckp.pt", type=str)#chatterbox_model_weight
    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image-size", default=768, type=int, help="image size")
    parser.add_argument("--model-max-length", default=2048, type=int)
    parser.add_argument("--lora-r", default=16, type=int)
    parser.add_argument("--vision-tower", default="/path/to/CLIP/clip-vit-large-patch14", type=str)
    parser.add_argument("--vision_tower_aux",default="/path/to/CLIP/clip-vit-large-patch14", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--coco2017_path", default="/path/to/MSCOCO2017/images/")#image
    parser.add_argument("--coco_val_path", default="/path/to/grouding_qa.json")#question and gt answer
    parser.add_argument("--save_out_path", default="/path/to/test_out.json")#predicted answer
    parser.add_argument('--pretrained', default="/path/to/Open-GroundingDino-main/groundingdino_swinb_cogcoor.pth",)
    return parser.parse_args(args)

def vision_branch_args():
    def get_args_parser():
        parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
        parser.add_argument('--config_file', '-c', default="./config/cfg_odvg_swinbase.py",type=str)
        parser.add_argument('--options',
            nargs='+',
            action=DictAction,
            help='override some settings in the used config, the key-value pair '
            'in xxx=yyy format will be merged into config file.')

        # dataset parameters
        parser.add_argument("--datasets", type=str, help='path to datasets json')
        parser.add_argument('--remove_difficult', action='store_true')
        parser.add_argument('--fix_size', action='store_true')

        # training parameters
        parser.add_argument('--output_dir', default='',
                            help='path where to save, empty for no saving')
        parser.add_argument('--note', default='',
                            help='add some notes to the experiment')
        parser.add_argument('--device', default='cuda',
                            help='device to use for training / testing')
        parser.add_argument('--seed', default=42, type=int)
        parser.add_argument('--resume', default='', help='resume from checkpoint')
        parser.add_argument('--pretrained', default="/path/to/Open-GroundingDino-main/groundingdino_swinb_cogcoor.pth",help='load from other checkpoint')
        parser.add_argument('--finetune_ignore', type=str, nargs='+')
        parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                            help='start epoch')
        parser.add_argument('--eval', action='store_true')
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--test', action='store_true')
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--find_unused_params', action='store_true')
        parser.add_argument('--save_results', action='store_true')
        parser.add_argument('--save_log', action='store_true')

        # distributed training parameters
        parser.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
        parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
        parser.add_argument('--rank', default=0, type=int,
                            help='number of distributed processes')
        parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
        parser.add_argument("--local-rank", type=int, help='local rank for DistributedDataParallel')
        parser.add_argument('--amp', action='store_true',
                            help="Train with mixed precision")
        parser.add_argument("--version", default="/path/to/llava-llama-2-13b-chat-lightning-preview")
        parser.add_argument("--weight", default="/path/to/chatterbox_grounding_ckp.pt",
                            type=str)  # chatterbox_model_weight
        parser.add_argument("--vision-tower", default="/path/to/CLIP/clip-vit-large-patch14", type=str)
        parser.add_argument("--vision_tower_aux", default="/path/to/CLIP/clip-vit-large-patch14", type=str)
        parser.add_argument("--coco2017_path", default="/path/to/MSCOCO2017/images/")  # image
        parser.add_argument("--coco_val_path", default="/path/to/grouding_qa.json")  # question and gt answer
        parser.add_argument("--save_out_path", default="/path/to/test_out.json")  # predicted answer
        return parser

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
    with open(save_json_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    return args

def preprocess(x: torch.Tensor) -> torch.Tensor:  # resize instead of padding
    # """Normalize pixel values and pad to a square input."""
    # # Normalize colors
    # x = (x - self.pixel_mean) / self.pixel_std

    # # # Pad
    # # h, w = x.shape[-2:]
    # # padh = self.img_size - h
    # # padw = self.img_size - w
    # # x = F.pad(x, (0, padw, 0, padh))

    x = x.float()
    x = torchvision.transforms.functional.normalize(x, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
    
    return x

def transform(x,size):
        trans = T.Compose([
            T.RandomResize([(size, size)])  # change to Resize?
        ])

        return trans(x, target=None)

def seg_prompt_for_bbox(prompt):
    question,bbox=prompt.split('?')
    bbox=eval(bbox)
    question=question+'?'
    return bbox,question

def get_coco_qa_list(path):
    """
    convert original question to fit chatterbox input, record other information(including image id, image path, gt bbox, etc)
    input:
        path:coco_val_path
    return:
        prompt_list:['Where is the toilet? [VG]'] (question for an image)
        file_name_list:['000000403385.jpg'] (filename for an image)
        gt_list:[[411.1, 237.7, 93.00999999999999, 242.3]]#left_x,left_y,w,h (gt bbox)
        object_name_list:['toilet'] (gt label)
        image_id_list:['403385'](image id)
    """
    prompt_list=[]
    file_name_list=[]
    gt_list=[]
    int_gt_list=[]
    object_name_list=[]
    answer_sent_list=[]
    image_id_list=[]
    with open(path,'r') as fr:
        file=json.load(fr)
        for f in file:
            image_id_list.append(f['image_id'])
            file_name_list.append(f['filename'])
            question=f['question']+' [VG]'
            prompt_list.append(question)
            answer=f['answer']
            sent,name_box=answer.split('<')
            sent+='.'
            answer_sent_list.append(sent)
            name_box=name_box.rstrip('>')
            name,box=name_box.split(':')
            bbox=eval(box)
            gt_list.append(bbox)
            int_gt_list.append([int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])])
            object_name_list.append(name)
    return prompt_list,file_name_list,gt_list,object_name_list,image_id_list


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[VG]")
    ret_token_idx = tokenizer("[VG]", add_special_tokens=False).input_ids
    args.vg_token_idx = ret_token_idx[0]  # 30523
    vision_args = vision_branch_args()
    model = JACK(
        args.local_rank,
        args.vg_token_idx,
        tokenizer,
        args.version,
        args.lora_r,
        args.precision,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        vision_tower=args.vision_tower,
        vision_tower_aux=args.vision_tower_aux,
        vision_branch_args=vision_args,
    )
  
    if args.weight:
        print('loading from ', args.weight)
        state_dict = torch.load(args.weight, map_location="cpu")['module']
        model.load_state_dict(state_dict, strict=True)
        
    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif args.precision == "fp16":
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
    else:
        model = model.float().cuda()

    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
    DEFAULT_IM_START_TOKEN = "<im_start>"
    DEFAULT_IM_END_TOKEN = "<im_end>"

    image_token_len = 256
    clip_image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)

    #Evaluate grouding
    prompt_list, file_name_list, gt_list, object_name_list, answer_sent_list = get_coco_qa_list(args.coco_val_path)#get questions
    output_list = []
    for idx in range(len(prompt_list)):
        output_dict = {}
        #annotation
        gt = gt_list[idx]
        output_dict['gt'] = gt
        output_dict['out_category'] = object_name_list[idx]
        image_path_or = os.path.join(args.coco2017_path,file_name_list[idx])
        output_dict['image_abs_path'] = image_path_or
        image_path = image_path_or
        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))
            continue

        #load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_image = image
        original_size_list = image.shape[:2]
        
        if args.precision == "bf16":
            images_clip = (
                clip_image_processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
                .unsqueeze(0)
                .cuda()
                .bfloat16()
            )
        elif args.precision == "fp16":
            images_clip = (
                clip_image_processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
                .unsqueeze(0)
                .cuda()
                .half()
            )
        else:
            images_clip = (
                clip_image_processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
                .unsqueeze(0)
                .cuda()
                .float()
            )
        images, _, _ = transform(Image.fromarray(ori_image),512)
        images = preprocess(torch.from_numpy(np.array(images)).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda().half()

        #generate conversation
        init_input = "Get Start"
        conversation_round = 0
        conv = get_default_conv_template("vicuna").copy()
        question = []
        answer = []

        #get input box
        region_list = []
        if init_input:
            conv.messages = []
            prompt = prompt_list[idx]
            output_dict['prompt'] = prompt
            if prompt == "EOS":
                break
            a = []
            b = []
            has_box = 0
            if len(a) == 0:#no box input
                region_list = region_list
            elif len(a) == len(b) and len(a)%2 == 0:
                w , h = image.shape[:2]
                last_box = None
                the_box = None
                for i in range(int(len(a)/2)):
                    input_box = [a[2*i],b[2*i],a[2*i+1],b[2*i+1]]
                    tensor_box = torch.tensor([input_box]) / torch.tensor([h, w, h, w])
                    tensor_box = tensor_box.half()
                    if last_box != None:
                        the_box = torch.cat((the_box,tensor_box),dim=0)
                    else:
                        the_box = tensor_box
                        last_box = 1
                if region_list == []:
                    region_list.append(the_box.cuda())
                else:
                    internal_box = torch.cat((region_list[0],the_box.cuda()),dim=0)
                    region_list = [internal_box]
                has_box = 1
            else:
                print("please input corret number of points")
                break
            if has_box:
                print(region_list)
            if "<bbox>" in prompt and len(a) == 0:
                print("please input box")
                break

            #convert token & format
            if conversation_round == 0:
                prompt = DEFAULT_IMAGE_TOKEN + " " + prompt
                replace_image_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
                replace_image_token = DEFAULT_IM_START_TOKEN + replace_image_token + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_image_token)
                conversation_round += 1
            else:
                conversation_round += 1
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], "")
            new_prompt = ""
            if conversation_round -1 > 0:
                prompt = conv.get_prompt(False)
                question.append(prompt)
                if len(question) > 1:
                    question[-2] = (
                    question[-2].replace("<bbox>", "").replace("[VG]", "")
                    )
                for i, num_round in enumerate(question):
                    if i < len(question) - 1 :
                        new_prompt += question[i]
                        new_prompt += answer[i]
                    else:
                        new_prompt += question[i]
                prompt = new_prompt  
            else:
                prompt = conv.get_prompt(True)
                question.append(prompt)

            #evaluate
            input_ids = tokenizer(prompt).input_ids
            input_ids = torch.LongTensor(input_ids).unsqueeze(0).cuda()
            output_ids,pred_box = model.evaluate(
                images_clip,
                images,
                region_list,
                input_ids,
                max_new_tokens=512,
            )

            text_output = tokenizer.decode(output_ids[0], skip_special_tokens=False)
            answer.append(text_output.split("ASSISTANT:")[-1].split("[VG]")[0])

            text_output = (
                text_output.replace(DEFAULT_IMAGE_PATCH_TOKEN, "")
                .replace("\n", "")
                .replace("  ", "")
            )
            print("text_output: ", text_output)

            # for grounding task
            if len(pred_box) > 0:
                box_threshold = 0.3
                logits = pred_box[-1]["pred_logits"].sigmoid()[0]  # (nq, 256)
                boxes = pred_box[-1]["pred_boxes"][0]  # (nq, 4)
                logits_filt = logits.cpu().clone()
                boxes_filt = boxes.cpu().clone()
                filt_mask = logits_filt.max(dim=1)[0] > box_threshold
                logits_filt = logits_filt[filt_mask]  # num_filt, 256
                boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
                bbox = []
                score = []
                for i, box in enumerate(boxes_filt):
                    box = box * torch.Tensor([original_size_list[1], original_size_list[0], original_size_list[1], original_size_list[0]])
                    # from xywh to xyxy
                    box[:2] -= box[2:] / 2
                    box[2:] += box[:2]
                    bbox.append([int(j.item()) for j in box[:]])
                    score.append(logits_filt[i].max().item())

                if len(score)==0:
                    score=[0]
                    bbox=[[0,0,1,1]]

                #find bbox with max score
                pos=score.index(max(score))
                pred_box=bbox[pos]
                pred_score=score[pos]
                output_dict['score'] = pred_score
                output_dict['out_boxes'] = pred_box
                print('predict box: ',pred_box)
            output_list.append(output_dict)

    #save prediction results
    os.makedirs(os.path.dirname(args.save_out_path), exist_ok=True)
    with open(args.save_out_path, 'w') as fw:
        json.dump(output_list, fw, indent=1)

if __name__ == "__main__":
    main(sys.argv[1:])
