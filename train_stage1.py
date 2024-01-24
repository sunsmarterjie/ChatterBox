import argparse
import os
import shutil
import sys
import time
from functools import partial

import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from torch.utils.tensorboard import SummaryWriter

from model.ChatterBox_Referrring_Grounding_grounding_dino import ChatterBox

from utils.chatterbox_dataset_grounding import GroundingDataset
from utils.chatterbox_dataset_grounding import collate_fn as grounding_collate_fn


from utils.utils import (
    AverageMeter,
    ProgressMeter,
    Summary,
    dict_to_cuda,
    intersectionAndUnionGPU,
)

import json
from utils.slconfig import DictAction, SLConfig


def parse_args(args):
    parser = argparse.ArgumentParser(description="ChatterBox Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="../llava-llama-2-13b-chat-lightning-preview"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        # default="bf16",
        default="fp16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=768, type=int, help="image size")
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument(
        "--vision-tower", default="../clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--dataset", default="jack||vqa", type=str
    )
    parser.add_argument("--vqa_data", default="../datasets/llava_instruct_150k", type=str)
    # parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str)
    parser.add_argument("--dataset_dir", default="../datasets/VG/",
                        type=str)
    parser.add_argument("--base_coco17_dir", default="../datasets/MSCOCO2017/",
                        type=str)
    parser.add_argument("--base_coco14_dir", default="../datasets/MSCOCO2014/",
                        type=str)
    parser.add_argument("--base_vg_dir", default="../datasets/VG/",
                        type=str)
    parser.add_argument("--base_flickr_dir", default="../datasets/flicker30k/",
                        type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="chatterbox", type=str)

    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--steps_per_epoch", default=3000, type=int)

    parser.add_argument("--grounding_batch_size", default=6, type=int, help="batch size per device per step")
    parser.add_argument("--referring_batch_size", default=6, type=int, help="batch size per device per step")
    parser.add_argument("--vqa_batch_size", default=6, type=int, help="batch size per device per step")
    parser.add_argument("--grounding_grad_accumulation_steps", default=20, type=int)
    parser.add_argument("--referring_grad_accumulation_steps", default=20, type=int)
    parser.add_argument("--vqa_grad_accumulation_steps", default=20, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.000030, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="PATH_TO_DINO", type=str)
    parser.add_argument("--weight", default="",type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
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
        parser.add_argument('--pretrained', default="../groundingdino_swinb_cogcoor.pth",help='load from other checkpoint')
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


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

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
      
    model = ChatterBox(
        args.local_rank,
        args.vg_token_idx,
        tokenizer,
        args.version,
        args.lora_r,
        args.precision,
        vision_tower=args.vision_tower,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        vision_branch_args=vision_args,
    )

    if vision_args.pretrained:
        state_dict = torch.load(vision_args.pretrained, map_location='cpu')['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'transformer.tgt_embed.weight' not in k and 'class' not in k:
                new_state_dict[k] = v
        msg = model.load_vision_dict(new_state_dict, strict=False)
        print('loading vision branch  >> ', msg)

    if args.weight:
        print('loading from ', args.weight)
        state_dict = torch.load(args.weight, map_location="cpu")['module']
        model.load_state_dict(state_dict, strict=True)

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1


    # print('before build dataset')
    grounding_dataset = GroundingDataset(
        args.dataset_dir,
        args.base_coco17_dir,
        tokenizer,
        args.vision_tower,
        dataset="refcocoground||cocoground||jackground||jacklogicground",
        sample_rate=[4, 4, 1, 1],
        vqa_data='llava_instruct_150k',
    )


    ds_grounding_config = {
        "train_micro_batch_size_per_gpu": args.grounding_batch_size,
        # "train_micro_batch_size_per_gpu": args.grounding_batch_size,
        "gradient_accumulation_steps": args.grounding_grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 50,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
            # "loss_scale": 0,
            # "loss_scale_window": 1000,
            # "hysteresis": 2,
            # "min_loss_scale": 1
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }


    model_engine, optimizer, grounding_dataloader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=grounding_dataset,
        collate_fn=partial(grounding_collate_fn, tokenizer=tokenizer),
        config=ds_grounding_config,
    )

    grounding_iter = iter(grounding_dataloader)

    for epoch in range(args.start_epoch, args.epochs):
        train(
            grounding_dataloader,
            model_engine,
            epoch,
            scheduler,
            writer,
            grounding_iter,
            optimizer,
            args,
        )


def train(
        grounding_dataloader,
        model,
        epoch,
        scheduler,
        writer,
        grounding_iter,
        optimizer,
        args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    vqa_losses = AverageMeter("VQALoss", ":.4f")
    vg_losses = AverageMeter("VGLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            vqa_losses,
            vg_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):

        for i in range(args.grounding_grad_accumulation_steps):
            try:
                input_dict = next(grounding_iter)
            except:
                grounding_iter = iter(grounding_dataloader)
                input_dict = next(grounding_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            try:
                if args.precision == "fp16":
                    input_dict["images"] = input_dict["images"].half()
                    input_dict["images_clip"] = input_dict["images_clip"].half()
                elif args.precision == "bf16":
                    input_dict["images"] = input_dict["images"].bfloat16()
                    input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
                else:
                    input_dict["images"] = input_dict["images"].float()
                    input_dict["images_clip"] = input_dict["images_clip"].float()
            except:
                continue

            meta_input_dict = {
                "images": input_dict['images'],
                "images_clip": input_dict['images_clip'],
                "regions_lists": [],
                'input_ids': input_dict['input_ids'],
                'labels': input_dict['labels'],
                'attention_masks': input_dict['attention_masks'],
                'offset': input_dict['offset'],
                'bboxes_gt_list': input_dict['bboxes_gt_list'],
                'label_gt_list': input_dict['label_list'],
            }


            # with torch.cuda.amp.autocast(enabled=True):
            output_dict = model(**meta_input_dict)

            vqa_loss = output_dict["vqa_loss"]
            vg_loss = output_dict["vg_loss"]
            loss = vqa_loss + vg_loss

            losses.update(loss.item(), input_dict["images"].size(0))
            vqa_losses.update(vqa_loss.item(), input_dict["images"].size(0))
            vg_losses.update(vg_loss.item(), input_dict["images"].size(0))
            model.backward(loss)
            model.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                vqa_losses.all_reduce()
                vg_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/vg_losses", vg_losses.avg, global_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            vqa_losses.reset()
            vg_losses.reset()

        if global_step != 0:
            scheduler.step()  # new add line
            curr_lr = scheduler.get_last_lr()
            print('curr_lr = ', curr_lr)
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

        if global_step % 50 == 0:
            save_dir = os.path.join('./outputs/', f'epoch_{epoch}')
            if args.local_rank == 0:
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model.save_checkpoint(save_dir)



if __name__ == "__main__":
    main(sys.argv[1:])
