import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
import torchvision

from .conversation import get_default_conv_template
from .utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
)
import re

import utils.transforms as T
from .box_ops import box_xyxy_to_cxcywh, box_xywh_to_cxcywh
from PIL import Image
import random

DEFAULT_REGION_TOKEN = "<region>"

img_size = 512


class JackLogicVQADataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = img_size
    region_size = 224

    def __init__(
            self,
            base_root,
            tokenizer,
            vision_tower,
            anno_path,
            samples_per_epoch=500 * 8 * 2 * 10,
            precision: str = "fp32",
            image_size: int = 224,
            num_classes_per_sample: int = 3,
            query_bbox_rate: float = 0.5,  # note to use this args or not ?
    ):
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        self.query_bbox_rate = query_bbox_rate

        self.base_root = base_root
        # self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        # self.transform = dino_transform  # transforms for dino detection
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.data_path = os.path.join(base_root)
        with open(os.path.join(anno_path, 'CB-LC.json')) as f:
            jack_json = json.load(f)
        self.jack_json = jack_json['data']

        self.replace_names = ['the region', 'this region']

        self.first_q = "This is an image. Can you answer the next questions about the specific regions in the image?  "
        self.first_a = "Sure, I will answer your questions.  "

    def __len__(self):
        return self.samples_per_epoch

    def transform(self, x):
        trans = T.Compose([
            T.RandomResize([(self.img_size, self.img_size)])  # change to Resize?
        ])

        return trans(x, target=None)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:  # resize instead of padding

        x = x.float()
        x = torchvision.transforms.functional.normalize(x, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

        return x

    def postprocess_bbox(self, bboxes_raw, ratios):

        h, w = self.img_size, self.img_size  # 图像的size变换 -> box的size变换
        boxes_gt = []
        for box in bboxes_raw:
            if len(box) == 0:  # this conversation has no bbox
                boxes_gt.append([])
                continue

            if isinstance(box, list):
                box = torch.tensor(box)

            ratio_width, ratio_height = ratios
            scaled_boxes = box * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])

            scaled_boxes = box_xyxy_to_cxcywh(scaled_boxes)
            scaled_boxes = scaled_boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            boxes_gt.append(scaled_boxes)
        return boxes_gt

    def strbbox2bbox(self, bboxes_str=None):
        if len(bboxes_str) == 0 or ";" in bboxes_str[0] or 'x1' in bboxes_str[0] or '?' in bboxes_str[0] or \
                '[0,0,0,0]' in bboxes_str[0] or '[]' in bboxes_str[0] or 'white and multi-storied and garage' in \
                bboxes_str[0] \
                or 'lap:[220, 151, 305]' in bboxes_str[0] or 'yellow and blue [equipment]' in bboxes_str[0]:
            return []
        bboxes_split_str = bboxes_str[0].split(']')[:-1]
        bboxes = []
        for bbox_split_str in bboxes_split_str:
            sta = bbox_split_str.find('[')
            bbox = list(eval(bbox_split_str[sta + 1:]))

            bboxes.append(bbox)
            if len(bbox) == 0:
                print(bboxes_str)
                assert False

        return bboxes

    def __getitem__(self, idx):

        while True:
            idx = random.randint(0, len(self.jack_json) - 1)
            image_path = os.path.join(self.data_path, self.jack_json[idx]['image'])
            img = cv2.imread(image_path)
            images_ori = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ori_size = images_ori.shape[:2]

            # preprocess images for clip
            images_clip = self.clip_image_processor.preprocess(images_ori, return_tensors="pt")[
                "pixel_values"
            ][0]
            # print('images_clip  >>> ', images_clip.shape)
            image_token_len = (images_clip.shape[1] // 14) * (
                    images_clip.shape[2] // 14
            )  # FIXME: 14 is hardcoded patch size

            images, _, ratios = self.transform(Image.fromarray(images_ori))  # preprocess images for dino, check this
            # resize = images.shape[:2]

            source = self.jack_json[idx]["conversation"]

            conv = get_default_conv_template(
                "vicuna"
            ).copy()  # conversation_lib.default_conversation.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

            conversations = []
            bboxes_human = []
            bboxes_gpt = []

            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]
            conv.messages = []

            for j, sentence in enumerate(source):  # note here: the model_max_length only contains about 6-7 VQAs
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{j}"

                if j % 2 == 0:

                    if '[it]' in sentence['value']:
                        sentence['value'] = sentence['value'].replace('[it]', 'it')

                    # 'the cup is on the desk. <cup:[238, 249, 298, 511], red and orange desk:[241, 289, 300, 390]>'
                    # extract the bboxes string: <cup:[238, 249, 298, 511], red and orange desk:[241, 289, 300, 390]>
                    bboxes_str = re.findall(r"<(.+?)>", sentence["value"])
                    # extract the bboxes : [[238, 249, 298, 511], [241, 289, 300, 390]]
                    bboxes = self.strbbox2bbox(bboxes_str)
                    # delete the bboxes string: 'the cup is on the desk.'
                    sentence['value'] = sentence['value'][:sentence['value'].find('<')]

                    bboxes_human.append([])

                    if j == 0:
                        sentence["value"] = '<image>\n' + ' ' + sentence["value"]  # put <image> in the most front

                elif j % 2 == 1:
                    if '<' in sentence['value']:
                        sentence['value'] = sentence['value'][:sentence['value'].find('<')]

                conv.append_message(role, sentence["value"])

            conversations.append(conv.get_prompt())

            questions = conversations
            sampled_classes = conversations

            # replace <image> token
            # region_token_len = 256
            for i in range(len(conversations)):
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
                replace_token = (
                        DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
                conversations[i] = conversations[i].replace(
                    DEFAULT_IMAGE_TOKEN, replace_token
                )

            images = self.preprocess(torch.from_numpy(np.array(images)).permute(2, 0, 1).contiguous())

            #### postprocess bbox
            bboxes_gpt = self.postprocess_bbox(bboxes_gpt, ratios)  # for DINO prediction

            if conversations[0].count("<im_start>") == 1:
                break

        return (
            images,
            images_clip,
            conversations,
            [],
            -1,
        )


class JackVQADataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = img_size
    region_size = 224

    def __init__(
            self,
            base_root,
            tokenizer,
            vision_tower,
            anno_path,
            samples_per_epoch=500 * 8 * 2 * 10,
            precision: str = "fp32",
            image_size: int = 224,
            num_classes_per_sample: int = 3,
            query_bbox_rate: float = 0.5,  # note to use this args or not ?
    ):
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        self.query_bbox_rate = query_bbox_rate

        self.base_root = base_root
        # self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        # self.transform = dino_transform  # transforms for dino detection
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.data_path = os.path.join(base_root)
        with open(os.path.join(anno_path, 'CB-MRG.json')) as f:
            jack_json = json.load(f)
        self.jack_json = jack_json['data']

        self.replace_names = ['the region', 'this region']

        self.first_q = "This is an image. Can you answer the next questions about the specific regions in the image?  "
        self.first_a = "Sure, I will answer your questions.  "

    def __len__(self):
        return self.samples_per_epoch

    def transform(self, x):
        trans = T.Compose([
            T.RandomResize([(self.img_size, self.img_size)])  # change to Resize?
        ])

        return trans(x, target=None)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:  # resize instead of padding

        x = x.float()
        x = torchvision.transforms.functional.normalize(x, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

        return x

    def postprocess_bbox(self, bboxes_raw, ratios):
        h, w = self.img_size, self.img_size  # 图像的size变换 -> box的size变换
        boxes_gt = []
        for box in bboxes_raw:
            if len(box) == 0:  # this conversation has no bbox
                boxes_gt.append([])
                continue

            if isinstance(box, list):
                box = torch.tensor(box)

            ratio_width, ratio_height = ratios
            scaled_boxes = box * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])

            scaled_boxes = box_xyxy_to_cxcywh(scaled_boxes)
            scaled_boxes = scaled_boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            boxes_gt.append(scaled_boxes)
        return boxes_gt

    def strbbox2bbox(self, bboxes_str=None):
        if len(bboxes_str) == 0 or ";" in bboxes_str[0] or 'x1' in bboxes_str[0] or '?' in bboxes_str[0] or \
                '[0,0,0,0]' in bboxes_str[0] or '[]' in bboxes_str[0] or 'white and multi-storied and garage' in \
                bboxes_str[0] \
                or 'lap:[220, 151, 305]' in bboxes_str[0] or 'yellow and blue [equipment]' in bboxes_str[0]:
            return []
        bboxes_split_str = bboxes_str[0].split(']')[:-1]
        bboxes = []
        for bbox_split_str in bboxes_split_str:
            sta = bbox_split_str.find('[')
            bbox = list(eval(bbox_split_str[sta + 1:]))

            bboxes.append(bbox)
            if len(bbox) == 0:
                print(bboxes_str)
                assert False

        return bboxes

    def __getitem__(self, idx):

        while True:
            idx = random.randint(0, len(self.jack_json) - 1)
            image_path = os.path.join(self.data_path, self.jack_json[idx]['image'])
            img = cv2.imread(image_path)
            images_ori = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ori_size = images_ori.shape[:2]

            # preprocess images for clip
            images_clip = self.clip_image_processor.preprocess(images_ori, return_tensors="pt")[
                "pixel_values"
            ][0]
            # print('images_clip  >>> ', images_clip.shape)
            image_token_len = (images_clip.shape[1] // 14) * (
                    images_clip.shape[2] // 14
            )  # FIXME: 14 is hardcoded patch size

            images, _, ratios = self.transform(Image.fromarray(images_ori))  # preprocess images for dino, check this
            # resize = images.shape[:2]

            source = self.jack_json[idx]["conversation"]

            conv = get_default_conv_template(
                "vicuna"
            ).copy()  # conversation_lib.default_conversation.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

            conversations = []
            bboxes_human = []
            bboxes_gpt = []

            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]
            conv.messages = []

            for j, sentence in enumerate(source):  # note here: the model_max_length only contains about 6-7 VQAs
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{j}"

                if j % 2 == 0:

                    # 'the cup is on the desk. <cup:[238, 249, 298, 511], red and orange desk:[241, 289, 300, 390]>'
                    # extract the bboxes string: <cup:[238, 249, 298, 511], red and orange desk:[241, 289, 300, 390]>
                    bboxes_str = re.findall(r"<(.+?)>", sentence["value"])
                    # extract the bboxes : [[238, 249, 298, 511], [241, 289, 300, 390]]
                    bboxes = self.strbbox2bbox(bboxes_str)
                    # delete the bboxes string: 'the cup is on the desk.'
                    sentence['value'] = sentence['value'][:sentence['value'].find('<')]

                    bboxes_human.append([])

                    if j == 0:
                        sentence["value"] = '<image>\n' + ' ' + sentence["value"]  # put <image> in the most front

                elif j % 2 == 1:

                    if '<' in sentence['value']:
                        sentence['value'] = sentence['value'][:sentence['value'].find('<')]

                conv.append_message(role, sentence["value"])

            conversations.append(conv.get_prompt())

            questions = conversations
            sampled_classes = conversations

            # replace <image> token
            # region_token_len = 256
            for i in range(len(conversations)):
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
                replace_token = (
                        DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
                conversations[i] = conversations[i].replace(
                    DEFAULT_IMAGE_TOKEN, replace_token
                )

            images = self.preprocess(torch.from_numpy(np.array(images)).permute(2, 0, 1).contiguous())

            #### postprocess bbox
            bboxes_gpt = self.postprocess_bbox(bboxes_gpt, ratios)  # for DINO prediction

            if conversations[0].count("<im_start>") == 1:
                break

        return (
            images,
            images_clip,
            conversations,
            [],
            -1,
        )


class LlavaDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = img_size

    def __init__(
            self,
            base_image_dir,
            tokenizer,
            vision_tower,
            anno_path,
            samples_per_epoch=500 * 8 * 2 * 10,  # the number should be reset
            precision: str = "fp32",
            image_size: int = 224,
            num_classes_per_sample: int = 3,
            # exclude_val=False,
            vqa_data="llava_instruct_150k",
    ):
        # self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        # self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        DATA_DIR = os.path.join(base_image_dir, "llava_dataset")
        self.vqa_image_root = os.path.join(base_image_dir, "train2017")
        with open(os.path.join(anno_path, "{}.json".format(vqa_data))) as f:
            vqa_data = json.load(f)
        self.vqa_data = vqa_data

        # print("vqa_data: ", len(self.vqa_data))

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:  # also resize like jack instead of padding ?
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def transform(self, x):
        trans = T.Compose([
            T.RandomResize([(self.img_size, self.img_size)])  # change to Resize?
        ])

        return trans(x, target=None)[0]

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.vqa_data) - 1)  # why reset idx here ?
        item = self.vqa_data[idx]

        image_path = os.path.join(self.vqa_image_root, item["image"])
        img = cv2.imread(image_path)
        images = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ori_size = images.shape[:2]
        images_clip = self.clip_image_processor.preprocess(images, return_tensors="pt")[
            "pixel_values"
        ][
            0
        ]  # preprocess images for clip -> what does this operation do (patch embedding) ?
        image_token_len = (images_clip.shape[1] // 14) * (
                images_clip.shape[2] // 14
        )  # FIXME: 14 is hardcoded patch size

        # images = self.transform.apply_image(images)  # preprocess images for sam
        images = self.transform(Image.fromarray(images))  # preprocess images for sam
        # resize = images.shape[:2]

        source = item["conversations"]
        conv = get_default_conv_template(
            "vicuna"
        ).copy()  # conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        conversations = []
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            if j % 2 == 1:
                sentence['value'] = sentence['value']  # + '[VG]'
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

        questions = conversations
        sampled_classes = conversations

        # replace <image> token
        for i in range(len(conversations)):
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversations[i] = conversations[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

        images = self.preprocess(torch.from_numpy(np.array(images)).permute(2, 0, 1).contiguous())

        return (
            images,
            images_clip,
            conversations,
            [],  # bbox
            -1,  # label
        )


def collate_fn(batch, tokenizer=None):
    images_list = []
    images_clip_list = []
    conversation_list = []
    bboxes_gt_list = []
    label_list = []
    offset_list = [0]
    cnt = 0
    for (
            images,
            images_clip,
            conversations,
            bbox,
            label,
    ) in batch:
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        bboxes_gt_list.append(bbox)
        cnt += len(conversations)
        offset_list.append(cnt)

    tokenize_data = tokenizer(
        conversation_list,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )

    input_ids = tokenize_data.input_ids
    attention_masks = tokenize_data.attention_mask

    IGNORE_TOKEN_ID = -100
    conv = get_default_conv_template("vicuna").copy()
    targets = input_ids.clone()
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID

            cur_len += round_len
        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    return {
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "attention_masks": attention_masks,
        "offset": torch.LongTensor(offset_list),
        "labels": targets,
        "label_list": label_list,
        "bboxes_gt_list": bboxes_gt_list,
    }


class VQADataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 768

    def __init__(
            self,
            base_image_dir,
            base_coco_dir,
            tokenizer,
            vision_tower,
            dataset="jack||vqa",
            sample_rate=[7, 3],
            vqa_data='llava_instruct_150k',
    ):
        dataset = "jacklogicvqa||jackvqa||vqa"
        sample_rate = [1, 2, 7]

        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.tokenizer = tokenizer

        self.datasets = dataset.split("||")

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "jacklogicvqa":
                self.all_datasets.append(
                    JackLogicVQADataset(
                        base_root='../datasets/VG/',
                        tokenizer=tokenizer,
                        vision_tower=vision_tower,
                        anno_path='../datasets/CB-300K/'
                    )
                )
            elif dataset == "jackvqa":
                self.all_datasets.append(
                    JackVQADataset(
                        base_root='../datasets/VG/',
                        tokenizer=tokenizer,
                        vision_tower=vision_tower,
                        anno_path='../datasets/CB-300K/'
                    )
                )
            elif dataset == "vqa":
                self.all_datasets.append(
                    LlavaDataset(
                        base_coco_dir,
                        tokenizer,
                        vision_tower,
                        anno_path='../datasets/'
                    )
                )

    def __len__(self):
        return 1000000

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]

        return data[0]
