import argparse
import os
import sys

import cv2
import glob
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, CLIPImageProcessor

from model.ChatterBox_Referrring_Grounding_grounding_dino import JACK
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.conversation import get_default_conv_template

import utils.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import random
import json
from utils.slconfig import DictAction, SLConfig
import torchvision

def parse_args(args):
    parser = argparse.ArgumentParser(description="JACK chat")
    parser.add_argument("--version", default="./llava-llama-2-13b-chat-lightning-preview")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument("--vision_pretrained", default="PATH_TO_DINO", type=str)
    parser.add_argument("--weight", default="./Jack_grounding_dino/output_refer_gnd_vqa_resume_v3/epoch_3/global_step11963/mp_rank_00_model_states.pt", type=str)
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
    parser.add_argument(
        
        "--vision-tower", default="./CLIP/clip-vit-large-patch14/", type=str
    )
    parser.add_argument(
        "--vision_tower_aux",default="./CLIP/clip-vit-large-patch14/", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
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
        parser.add_argument('--pretrained', default="./Open-GroundingDino-main/groundingdino_swinb_cogcoor.pth",help='load from other checkpoint')
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
        return parser

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    # if args.rank == 0:
    save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
    # cfg.dump(save_cfg_path)
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
    # bbox,question2=bbox1.split('>')
    # bbox='['+bbox+']'
    bbox=eval(bbox)
    question=question+'?'
    return bbox,question

def get_list(path):
    prompt_list=[]
    file_name_list=[]
    image_id_list=[]
    gt_list=[]
    ref_id_list = []
    with open(path,'r') as fr:
        file=json.load(fr)
    for f in file:
        prompt_list.append(f['question'])
        file_name_list.append(f['file_name'])
        gt_list.append(f['answer'])
        image_id_list.append(f['image_id'])
        ref_id_list.append(f['ref_id'])
    return prompt_list,file_name_list,gt_list,image_id_list,ref_id_list

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
        # state_dict = torch.load(args.weight, map_location="cpu")
        # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # print(state_dict.keys())
        # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
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
    
    
    refcocog_test_path = "./evaluation/referring/gt/refcocog/test.json"
    save_out_path = './evaluation/referring/output/refcocog_test.json'
    prompt_list,image_path_list,gt_list,image_list,ref_id_list=get_list(refcoco_testb_path)
    # id_list,image_id_list,category_id_list,question_list,answer_list=get_list(refcoco_testb_path)
    cnt=0
    cnt+=1
    output_list=[]
    refcoco_testb_path = refcocog_test_path
    for idx in range(len(image_path_list)):
        output_dict = {}
        gt=gt_list[idx]
        image_id=image_list[idx]
        ref_id = ref_id_list[idx]
        output_dict['gt']=gt
        output_dict['ref_id'] = ref_id
        output_dict['id']=image_id
        coco2014_path="./datasets/MSCOCO2014/images/train2014/"
        image_name,jpg=image_path_list[idx].split('.')
        image_name_list=image_name.split('_')
        image_name2=''
        for i in range(len(image_name_list)-1):
            image_name2 += image_name_list[i]
            if i < len(image_name_list)-2:
                image_name2 += '_'
        image_name2+='.'
        image_name2+=jpg
        #image_path_or=os.path.join(coco2014_path,image_path_list[idx])
        image_path_or=os.path.join(coco2014_path,image_name2)
        output_dict['image_abs_path']=image_path_or
        image_path = image_path_or
        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))
            continue

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
        # images = transform.apply_image(image)
        images, _, _ = transform(Image.fromarray(ori_image),512)
        images = preprocess(torch.from_numpy(np.array(images)).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda().half()
        # images = transform(Image.fromarray(image), 512)
        # resize_list = [images.shape[:2]]
        resize_list = []
        init_input = "Get Start"
        conversation_round = 0
        conv = get_default_conv_template("vicuna").copy()
        question = []
        answer = []
        #while init_input:
        if init_input:
            conv.messages = []

            #prompt = input("Input EOS to change Image. Please input your prompt: ")
            prompt=prompt_list[idx] 
            # prompt=question_list[idx]
            bbox,prompt=seg_prompt_for_bbox(prompt)
            output_dict['input_bbox'] = bbox
            output_dict['prompt'] = prompt 
            # print(prompt)
            if prompt == "EOS":
                break
            img = cv2.imread(image_path)
            # a = []
            # b = []
            a=[bbox[0][0],bbox[0][2]]
            b=[bbox[0][1],bbox[0][3]]

            def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    xy = "%d,%d" % (x, y)
                    a.append(x)
                    b.append(y)
                    cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
                    cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                                1.0, (0, 0, 0), thickness=1)
                    cv2.imshow("image", img)
                    print(x,y)


            # cv2.namedWindow("image")
            # cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
            # cv2.imshow("image", img)
            # cv2.resizeWindow("image", img.shape[1] + 50, img.shape[0] + 50)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            has_box = 0
            region_list = []
            if len(a) == 0:
                region_list = []
            elif len(a) == 2:
                w , h = image.shape[:2]

                input_box = [[a[0],b[0],a[1],b[1]]]
                # region_list.append(torch.tensor(input_box) / torch.tensor([w, h, w, h], dtype=torch.half))
                tensor_box = torch.tensor(input_box) / torch.tensor([[h, w, h, w]])
                tensor_box = tensor_box.half()
                region_list.append(tensor_box.cuda())
                # region_list = [[b[0],a[0],b[1],a[1]]]
                has_box = 1
            else:
                print("can only input 2 points")
                break

            if has_box and len(region_list) > 1:
                print("Only one box can be input now, please retry!")
                break
            if has_box:
                print(region_list)
                # regions.append(extract_regions([region_list], torch.from_numpy(ori_image),clip_image_processor_aux)[0])
            if conversation_round == 0:
                prompt = DEFAULT_IMAGE_TOKEN + " " + prompt + 'Please give a brief answer in one sentence less than 10 words.'
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

            # print(prompt)
            # print(question)
            # print(answer)

            input_ids = tokenizer(prompt).input_ids
            input_ids = torch.LongTensor(input_ids).unsqueeze(0).cuda()

            output_ids,pred_box = model.evaluate(
                images_clip,
                images,
                region_list,
                input_ids,
                max_new_tokens=320,
            )



            text_output = tokenizer.decode(output_ids[0], skip_special_tokens=False)
            answer.append(text_output.split("ASSISTANT:")[-1].split("[VG]")[0])

            text_output = (
                text_output.replace(DEFAULT_IMAGE_PATCH_TOKEN, "")
                .replace("\n", "")
                .replace("  ", "")
            )
            print("text_output: ", text_output)
            save_text_output=text_output.split('ASSISTANT:')[-1]
            save_text_output=save_text_output.split('</s>')[0]
            print("answer:",save_text_output)
            output_dict['text_output']=save_text_output
            print('+++++++++++++++++++++++++++++++++++++++')
            print(len(output_list))
            print('+++++++++++++++++++++++++++++++++++++++')
            output_list.append(output_dict)
    with open(save_out_path,'w') as fw:
        json.dump(output_list,fw,indent=1)
if __name__ == "__main__":
    main(sys.argv[1:])
