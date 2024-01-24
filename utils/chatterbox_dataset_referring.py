import copy
import os
import random
import numpy as np
import torch
from transformers import CLIPImageProcessor
import torchvision

import os

#from gpt4roi.train.train import preprocess, preprocess_multimodal
from mmdet.datasets import CocoDataset
from mmdet.datasets.api_wrappers import COCO
import random
import json
import cv2
from PIL import Image
import re

import utils.transforms as T
from .box_ops import box_xyxy_to_cxcywh, box_xywh_to_cxcywh

from .conversation import get_default_conv_template
from .utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
)

IGNORE_INDEX = -100

QUESTIONS = [
    '<spi_descript>'
    # "Could you please help me recognize the <spi_descript> in this picture?",
    # "Can you assist me in identifying the <spi_descript> in this image?",
    # "Can you tell what is at <spi_descript> in this image?",
    # "What is the object located in <spi_descript> in this picture?",
    # "Would you be able to tell me what is at <spi_descript> in this image?",
    # "Could you identify the item in <spi_descript> for me in this photograph?",
    # "Can you help me recognize the object in <spi_descript> in this picture?",
    # "I'm trying to figure out what is located in <spi_descript> in this image, could you help me?",
    # "What object can you see at <spi_descript> in this photograph?",
    # "Would you mind telling me what is located in <spi_descript> in this picture?",
    # "Can you assist me in identifying the item at <spi_descript> in this image?",
    # "What is the thing that can be seen in <spi_descript> in this photograph?",
    # "Could you please help me identify the object in <spi_descript> in this picture?"
]

QUESTIONS_RefCOCO = [
    'Can you provide me with a detailed description of the region in the picture marked by <spi_descript>?',
    "I'm curious about the region represented by <spi_descript> in the picture. Could you describe it in detail?",
    'What can you tell me about the region indicated by <spi_descript> in the image?',
    "I'd like to know more about the area in the photo labeled <spi_descript>. Can you give me a detailed description?",
    'Could you describe the region shown as <spi_descript> in the picture in great detail?',
    'What details can you give me about the region outlined by <spi_descript> in the photo?',
    'Please provide me with a comprehensive description of the region marked with <spi_descript> in the image.',
    'Can you give me a detailed account of the region labeled as <spi_descript> in the picture?',
    "I'm interested in learning more about the region represented by <spi_descript> in the photo. Can you describe it in detail?",
    'What is the region outlined by <spi_descript> in the picture like? Could you give me a detailed description?',
    'Can you provide me with a detailed description of the region in the picture marked by <spi_descript>, please?',
    "I'm curious about the region represented by <spi_descript> in the picture. Could you describe it in detail, please?",
    'What can you tell me about the region indicated by <spi_descript> in the image, exactly?',
    "I'd like to know more about the area in the photo labeled <spi_descript>, please. Can you give me a detailed description?",
    'Could you describe the region shown as <spi_descript> in the picture in great detail, please?',
    'What details can you give me about the region outlined by <spi_descript> in the photo, please?',
    'Please provide me with a comprehensive description of the region marked with <spi_descript> in the image, please.',
    'Can you give me a detailed account of the region labeled as <spi_descript> in the picture, please?',
    "I'm interested in learning more about the region represented by <spi_descript> in the photo. Can you describe it in detail, please?",
    'What is the region outlined by <spi_descript> in the picture like, please? Could you give me a detailed description?',
]

REGION_QUESTIONS = [
    'Which part of your overall description corresponds to the specific area of the image <spi_descript> you are referring to?',
    'In your initial description, which part corresponds to the particular area of the image <spi_descript> you are indicating?',
    'Can you specify which aspect of your overall description corresponds to the particular section of the image <spi_descript> you are pointing to?',
    'Which specific details from your overall description correspond to the particular area of the image <spi_descript> you are identifying?',
    'From your initial description, which parts specifically match the area of the image <spi_descript> you are referring to?',
    'Could you indicate which elements from your overall description relate to the particular section of the image <spi_descript> you are highlighting?',
    'Which aspects of your description correspond to the specific area of the image <spi_descript> you are referencing?',
    'Can you point out the specific parts of your description that correspond to the area of the image <spi_descript> you are focusing on?',
    'In your description, which details correspond to the specific portion of the image <spi_descript> you are indicating?',
    'Could you identify the specific parts of your description that match the section of the image <spi_descript> you are referring to?'
]

FINAL_QUESTIONS = [
    'Could you please give me a detailed description of these areas <spi_descript>?',
    'Can you provide a thorough description of the regions <spi_descript> in this image?',
    'Please describe in detail the contents of the boxed areas <spi_descript>.',
    'Could you give a comprehensive explanation of what can be found within <spi_descript> in the picture?',
    'Could you give me an elaborate explanation of the <spi_descript> regions in this picture?',
    'Can you provide a comprehensive description of the areas identified by <spi_descript> in this photo?',
    'Help me understand the specific locations labeled <spi_descript> in this picture in detail, please.',
    'What is the detailed information about the areas marked by <spi_descript> in this image?',
    'Could you provide me with a detailed analysis of the regions designated <spi_descript> in this photo?',
    'What are the specific features of the areas marked <spi_descript> in this picture that you can describe in detail?',
    'Could you elaborate on the regions identified by <spi_descript> in this image?',
    'What can you tell me about the areas labeled <spi_descript> in this picture?',
    'Can you provide a thorough analysis of the specific locations designated <spi_descript> in this photo?',
    'I am interested in learning more about the regions marked <spi_descript> in this image. Can you provide me with more information?',
    'Could you please provide a detailed description of the areas identified by <spi_descript> in this photo?',
    'What is the significance of the regions labeled <spi_descript> in this picture?',
    'I would like to know more about the specific locations designated <spi_descript> in this image. Can you provide me with more information?',
    'Can you provide a detailed breakdown of the regions marked <spi_descript> in this photo?',
    'What specific features can you tell me about the areas identified by <spi_descript> in this picture?',
    'Could you please provide a comprehensive explanation of the locations labeled <spi_descript> in this image?',
    'Can you provide a detailed account of the regions designated <spi_descript> in this photo?',
    'I am curious about the areas marked <spi_descript> in this picture. Can you provide me with a detailed analysis?',
    'What important details can you tell me about the specific locations identified by <spi_descript> in this image?',
    'Could you please provide a detailed description of the regions labeled <spi_descript> in this photo?',
    'What can you tell me about the features of the areas designated <spi_descript> in this picture?',
    'Can you provide a comprehensive overview of the regions marked <spi_descript> in this image?',
    'I would like to know more about the specific locations identified by <spi_descript> in this photo. Can you provide me with more information?',
    'What is the detailed information you have on the areas labeled <spi_descript> in this picture?',
    'Could you provide me with a thorough analysis of the regions designated <spi_descript> in this image?',
    'Can you provide a detailed explanation of the specific locations marked by <spi_descript> in this photo?'
]

REFG_QUESTIONS = [
    'Can you provide me with a detailed description of the region in the picture marked by <spi_descript>?',
    "I'm curious about the region represented by <spi_descript> in the picture. Could you describe it in detail?",
    'What can you tell me about the region indicated by <spi_descript> in the image?',
    "I'd like to know more about the area in the photo labeled <spi_descript>. Can you give me a detailed description?",
    'Could you describe the region shown as <spi_descript> in the picture in great detail?',
    'What details can you give me about the region outlined by <spi_descript> in the photo?',
    'Please provide me with a comprehensive description of the region marked with <spi_descript> in the image.',
    'Can you give me a detailed account of the region labeled as <spi_descript> in the picture?',
    "I'm interested in learning more about the region represented by <spi_descript> in the photo. Can you describe it in detail?",
    'What is the region outlined by <spi_descript> in the picture like? Could you give me a detailed description?',
    'Can you provide me with a detailed description of the region in the picture marked by <spi_descript>, please?',
    "I'm curious about the region represented by <spi_descript> in the picture. Could you describe it in detail, please?",
    'What can you tell me about the region indicated by <spi_descript> in the image, exactly?',
    "I'd like to know more about the area in the photo labeled <spi_descript>, please. Can you give me a detailed description?",
    'Could you describe the region shown as <spi_descript> in the picture in great detail, please?',
    'What details can you give me about the region outlined by <spi_descript> in the photo, please?',
    'Please provide me with a comprehensive description of the region marked with <spi_descript> in the image, please.',
    'Can you give me a detailed account of the region labeled as <spi_descript> in the picture, please?',
    "I'm interested in learning more about the region represented by <spi_descript> in the photo. Can you describe it in detail, please?",
    'What is the region outlined by <spi_descript> in the picture like, please? Could you give me a detailed description?',
]


class CocoDet(CocoDataset):

    def __init__(self,
                 tokenizer,
                 multimodal_cfg=None,
                 vis_processor=None,
                 vis_root='/home/TianYunjie/Workspace/datasets/MSCOCO2017',
                 add_eos=True,
                 ignore_instruction=True,
                 filter_small=False,
                 test_mode=False,
                 max_gt_per_img=100,
                 ):
        self.multimodal_cfg = multimodal_cfg
        self.tokenizer = tokenizer
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.max_gt_per_img = max_gt_per_img
        self.add_eos = add_eos
        self.ignore_instruction = ignore_instruction
        self.filter_small = filter_small
        self.test_mode = test_mode

        img_norm_cfg = dict(
            mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
            std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255], to_rgb=True)

        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(224, 224), keep_ratio=False),
            dict(type='RandomShift', shift_ratio=0.5, max_shift_px=32),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(2.0, 2.0)),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=224),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]

        test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(224, 224), keep_ratio=False),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(2.0, 2.0)),
            dict(type='RandomFlip', flip_ratio=0.),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=224),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]
        
        if test_mode:
            pipeline = test_pipeline
        else:
            pipeline = train_pipeline

        if test_mode:
            ann_file = f'{self.vis_root}/annotations/instances_val2017.json'
            img_prefix = self.vis_root + '/val2017'
        else:
            ann_file = f'{self.vis_root}/annotations/instances_train2017.json'
            img_prefix = self.vis_root + '/train2017'

        train = dict(
            ann_file=ann_file,
            img_prefix=img_prefix,
            test_mode=False,
            pipeline=pipeline)
        super(CocoDataset, self).__init__(**train)
        # TODO filter the small image? < 32 ?
        self.num_classes = len(self.CLASSES)
        begin_str = '<image>\nIn the conversation below, you simply answer the category name based on what you see ' \
                    'in the imagery inside a particular region.I will give you only one region each time. ' \
                    'Categories Containing '
        class_str = ', '.join(self.CLASSES)
        self.begin_str = begin_str + class_str + '.\n'
    
        self.sentence_heads = [
            'This is a ',
            "It is a ",
        ]
        self.sentence_tails =[
            " in the region.",
            ".",
            " in this region."
        ]

    def train_process_test(self, data_item):
        image = data_item['img'].data
        ori_labels = data_item['gt_labels'].data
        ori_bboxes = data_item['gt_bboxes'].data

        shuffle_ids = torch.randperm(len(ori_labels))
        if len(shuffle_ids) > self.max_gt_per_img:
            shuffle_ids = shuffle_ids[:self.max_gt_per_img]
        ori_bboxes = ori_bboxes[shuffle_ids]
        ori_labels = ori_labels[shuffle_ids]

        conversations = []

        conv = get_default_conv_template(
                "vicuna"
        ).copy()  # conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        conv.messages = []

        for i in range(len(ori_labels)):
            question = random.choice(QUESTIONS).strip()
            question = question.replace('<spi_descript>', '<bbox>')
            if i == 0:
                question = self.begin_str + question
            answer = self.CLASSES[ori_labels[i]]
            conv.append_message(roles["human"], question)
            answer = random.sample(self.sentence_heads, 1)[0] + answer + random.sample(self.sentence_tails, 1)[0]
            conv.append_message(roles["gpt"], answer)
        
        conversations.append(conv.get_prompt())

        image_token_len = (image.shape[1] // 14) * (image.shape[2] // 14)

        for i in range(len(conversations)):
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversations[i] = conversations[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

        data_dict = {
            'conversations': conversations
        }


        data_dict['image'] = image
        ori_bboxes = copy.deepcopy(ori_bboxes) / image.shape[1]

        data_dict['bboxes'] = ori_bboxes
        data_dict['img_metas'] = data_item['img_metas'].data
        return data_dict


    def process_text(self, data_item):
        if isinstance(data_item['img'], list):
            # test model
            data_item = {k: v[0] for k, v in data_item.items()}


        return self.train_process_test(data_item)


    def tokenize(self, text):
        res = self.tokenizer(
            text['instruction'] + text['answer'],
            return_tensors=None,
            padding='do_not_pad',
            truncation=True,
            max_length=512,
        )

        # manually add eos token
        if res['input_ids'][-1] != self.tokenizer.eos_token_id and len(
                res['input_ids']) < 512 and self.add_eos:
            res['input_ids'].append(self.tokenizer.eos_token_id)
            res['attention_mask'].append(1)
        labels = copy.deepcopy(res['input_ids'])
        # ignore instruction_token
        if self.ignore_instruction:
            bbox_index = labels.index(self.tokenizer.encode('<bbox>')[1])
            labels[:bbox_index] = [-100] * bbox_index

        res.update(labels=labels)
        return res

    def __getitem__(self, idx):
        idx = random.randint(0, len(self) - 1)  # new add

        data_item = super().__getitem__(idx)

        # img , input_ids, labels
        data_dict = self.process_text(data_item=data_item)

        return (
            data_dict['conversations'],
            data_dict['image'],
            data_dict['bboxes'],
            data_dict['img_metas']
        )


class RefCOCO(CocoDataset):
    CLASSES = ('object',)

    def __init__(self,
                 tokenizer,
                 multimodal_cfg=None,
                 vis_processor=None,
                 ann_file='./refcoco/instances.json',
                 img_prefix=None,
                 add_eos=True,
                 ignore_instruction=True,
                 filter_small=False,
                 test_mode=False,
                 max_gt_per_img=15,
                 ):

        self.multimodal_cfg = multimodal_cfg
        self.tokenizer = tokenizer
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.vis_processor = vis_processor
        self.max_gt_per_img = max_gt_per_img
        self.add_eos = add_eos
        self.ignore_instruction = ignore_instruction
        self.filter_small = filter_small
        self.test_mode = test_mode

        img_norm_cfg = dict(
            mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
            std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
            to_rgb=True)

        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(224, 224), keep_ratio=False),
            # dict(type='RandomShift', shift_ratio=0.5, max_shift_px=32),
            dict(type='FilterAnnotationsFlickr', min_gt_bbox_wh=(2.0, 2.0)),
            dict(type='RandomFlip', flip_ratio=0.),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='DefaultFormatBundleFlickr'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]

        test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(224, 224), keep_ratio=False),
            dict(type='FilterAnnotationsFlickr', min_gt_bbox_wh=(2.0, 2.0)),
            dict(type='RandomFlip', flip_ratio=0.),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=224),
            dict(type='DefaultFormatBundleFlickr'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]

        if test_mode:
            pipeline = test_pipeline
        else:
            pipeline = train_pipeline

        if test_mode:
            ann_file = self.ann_file
            img_prefix = self.img_prefix
        else:
            ann_file = self.ann_file
            img_prefix = self.img_prefix
        train = dict(
            ann_file=ann_file,
            img_prefix=img_prefix,
            test_mode=False,
            pipeline=pipeline, )
        super(CocoDataset, self).__init__(**train)
        # TODO filter the small image? < 32 ?
        self.num_classes = len(self.CLASSES)
        self.id_cap_dict = dict()
        self.begin_str = '<image>\n I will provide you with only one region ' \
                         'containing only one object, although there may be other ' \
                         'objects present in the image. It is recommended that you ' \
                         "describe the object's relative position with respect to other " \
                         'objects in the image, as well as its position within ' \
                         'the image and its basic attributes.'
        self.sentence_heads = [
            "Sure, there is a ",
            "OK, it is a ",
            "There is a ",
            "It is a ",
        ]
        self.sentence_tails =[
            " in the region.",
            ".",
            " in this region."
        ]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # TODO: obtain images that contain annotation
        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        num_remove_images = 0
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            if len(info['caption'].split(' ')) < 3:
                num_remove_images += 1
                continue
            info['filename'] = info['file_name']#.split('_')[-1]
            # convert data type for flickr
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])

            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        print(f'Filtered {num_remove_images} from  {self.ann_file} ')
        return data_infos

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """

        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        img_path = os.path.join(self.img_prefix, img_info['file_name'].split('_')[-1])
        self.id_cap_dict[img_info['file_name'].split('_')[-1]] = img_info['caption']
        # flickr
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue

            bbox = [x1, y1, x1 + w, y1 + h]

            gt_bboxes.append(bbox)
            gt_labels.append(img_info['caption'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            caption=img_info['caption'],
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def process_text(self, data_item):
        if isinstance(data_item['img'], list):
            # test model
            data_item = {k: v[0] for k, v in data_item.items()}

        return self.train_process_test(data_item)

    def train_process_test(self, data_item):
        image = data_item['img'].data
        ori_labels = data_item['gt_labels']
        ori_bboxes = data_item['gt_bboxes'].data

        sources = {'conversations': []}

        # DETAILS QUESTION

        shuffle_ids = torch.randperm(len(ori_labels))
        if len(shuffle_ids) > self.max_gt_per_img:
            shuffle_ids = shuffle_ids[:self.max_gt_per_img]
        select_bboxes = ori_bboxes[shuffle_ids]
        select_labels = [ori_labels[i] for i in shuffle_ids]


        conversations = []

        conv = get_default_conv_template(
                "vicuna"
        ).copy()  # conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        conv.messages = []

        for i in range(len(select_labels)):
            question = random.choice(QUESTIONS).strip()
            question = question.replace('<spi_descript>', ' <bbox>')
            answer = select_labels[i]  # already string
            # sources['conversations'].append({'from': 'human', 'value': question})
            # sources['conversations'].append({'from': 'gpt', 'value': answer})
            if i == 0:
                question = self.begin_str + question
            conv.append_message(roles["human"], question)
            answer = random.sample(self.sentence_heads, 1)[0] + answer + random.sample(self.sentence_tails, 1)[0]
            conv.append_message(roles["gpt"], answer)
        
        conversations.append(conv.get_prompt())

        # print('RefCOCO before  >>>> ', conversations)

        image_token_len = (image.shape[1] // 14) * (image.shape[2] // 14)

        for i in range(len(conversations)):
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversations[i] = conversations[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

        data_dict = {
            'conversations': conversations
        }

        data_dict['image'] = image

        # double for last detail question
        ori_bboxes = select_bboxes

        # print('refcoco ori_bboxes   >>> ', ori_bboxes)
        ori_bboxes = copy.deepcopy(ori_bboxes) / image.shape[1]

        data_dict['bboxes'] = ori_bboxes
        data_dict['img_metas'] = data_item['img_metas'].data
        return data_dict

    def __getitem__(self, idx):
        idx = random.randint(0, len(self) - 1)  # new add

        data_item = super().__getitem__(idx)
        max_loops = 10
        i = 0

        while True:
            if i > max_loops:
                raise ValueError('No gt_labels')
            i += 1
            if len(data_item['gt_labels']) == 0:
                idx = random.randint(0, len(self) - 1)
                print('    idx     >>>>', idx, len(self))
                data_item = super().__getitem__(idx)
            else:
                break
        data_dict = self.process_text(data_item=data_item)

        # return data_dict.values()
        return (
            data_dict['conversations'],
            data_dict['image'],
            data_dict['bboxes'],
            data_dict['img_metas']
        )


class RefCOCOP(RefCOCO):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.begin_str = '<image>\n I will provide you with only one region ' \
                         'containing only one object, although there may be other ' \
                         'objects present in the image. It is recommended that you ' \
                         "describe the object's relative position with respect to other " \
                         'objects in the image and its basic attibuts, you should not ' \
                         'give its position within the image' \


class RefCOCOG(RefCOCO):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.begin_str = """The <image> provides an overview of the picture.\n"""
    def train_process_test(self, data_item):
        image = data_item['img'].data
        ori_labels = data_item['gt_labels']
        ori_bboxes = data_item['gt_bboxes'].data


        shuffle_ids = torch.randperm(len(ori_labels))
        if len(shuffle_ids) > self.max_gt_per_img:
            shuffle_ids = shuffle_ids[:self.max_gt_per_img]
        select_bboxes = ori_bboxes[shuffle_ids]
        select_labels = [ori_labels[i] for i in shuffle_ids]

        conversations = []

        conv = get_default_conv_template(
                "vicuna"
        ).copy()  # conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        conv.messages = []

        for i in range(len(select_labels)):
            question = random.choice(REFG_QUESTIONS).strip()
            question = question.replace('<spi_descript>', f'region{i+1} <bbox>')
            answer = select_labels[i]  # already string
            if i == 0:
                question = self.begin_str + question
            conv.append_message(roles["human"], question)
            answer = random.sample(self.sentence_heads, 1)[0] + answer + random.sample(self.sentence_tails, 1)[0]
            conv.append_message(roles["gpt"], answer)
        
        conversations.append(conv.get_prompt())

        image_token_len = (image.shape[1] // 14) * (image.shape[2] // 14)

        for i in range(len(conversations)):
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversations[i] = conversations[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

        data_dict = {
            'conversations': conversations
        }

        data_dict['image'] = image

        # double for last detail question
        ori_bboxes = select_bboxes

        # print('refcocog ori_bboxes   >>> ', ori_bboxes)
        ori_bboxes = copy.deepcopy(ori_bboxes) / image.shape[1]

        data_dict['bboxes'] = ori_bboxes
        data_dict['img_metas'] = data_item['img_metas'].data
        return data_dict


class Flickr30k(CocoDataset):
    CLASSES = ('object',)
    def __init__(self,
                 tokenizer,
                 multimodal_cfg=None,
                 vis_processor=None,
                 ann_file='./OpenSource/final_flickr_separateGT_train.json',
                 img_prefix=None,
                 add_eos=True,
                 ignore_instruction=True,
                 filter_small=False,
                 test_mode=False,
                 max_gt_per_img=150,
                 ):

        self.multimodal_cfg = multimodal_cfg
        self.tokenizer = tokenizer
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.vis_processor = vis_processor
        # remove this
        self.max_gt_per_img = max_gt_per_img
        self.add_eos = add_eos
        self.ignore_instruction = ignore_instruction
        self.filter_small = filter_small
        self.test_mode = test_mode

        img_norm_cfg = dict(
            mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
            std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
            to_rgb=True)

        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(224, 224), keep_ratio=False),
            # dict(type='RandomShift', shift_ratio=0.5, max_shift_px=32),
            dict(type='FilterAnnotationsFlickr', min_gt_bbox_wh=(2.0, 2.0)),
            dict(type='RandomFlip', flip_ratio=0.),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='DefaultFormatBundleFlickr'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]

        test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(224, 224), keep_ratio=False),
            dict(type='FilterAnnotationsFlickr', min_gt_bbox_wh=(2.0, 2.0)),
            dict(type='RandomFlip', flip_ratio=0.),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=224),
            dict(type='DefaultFormatBundleFlickr'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]

        if test_mode:
            pipeline = test_pipeline
        else:
            pipeline = train_pipeline

        if test_mode:
            ann_file = self.ann_file
            img_prefix = self.img_prefix
        else:
            ann_file = self.ann_file
            img_prefix = self.img_prefix
        train = dict(
                ann_file=ann_file,
                img_prefix=img_prefix,
                test_mode=False,
                pipeline=pipeline,)
        super(CocoDataset, self).__init__(**train)
        # TODO filter the small image? < 32 ?
        self.num_classes = len(self.CLASSES)
        self.id_cap_dict = dict()
        self.begin_str = """The <image> provides an overview of the picture.\n"""
    
        self.sentence_heads = [
            "Sure, there is the ",
            "OK, it is the ",
            "There is the ",
            "It is the ",
        ]
        self.sentence_tails =[
            " in the region.",
            ".",
            " in this region."
        ]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # TODO: obtain images that contain annotation
        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            # convert data type for flickr
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])

            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        self.id_cap_dict[img_info['file_name']] = img_info['caption']
        # flickr
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id']  in self.cat_ids:
                pass
            else:
                raise ValueError('category_id not in self.cat_ids')
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                # flickr label
                gt_list = [img_info['caption'][atp[0]:atp[1]] for atp in ann['tokens_positive']]
                # TODO
                gt_labels.append(gt_list[0])  # TODO: one box might correspond to multiple labels, join with `, `
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            caption=img_info['caption'],
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def process_text(self, data_item):
        if isinstance(data_item['img'], list):
            # test model
            data_item = {k: v[0] for k, v in data_item.items()}

        return self.train_process_test(data_item)

    def train_process_test(self, data_item):
        image = data_item['img'].data
        ori_labels = data_item['gt_labels']
        ori_bboxes = data_item['gt_bboxes'].data

        # sources = {'conversations': []}

        conversations = []

        conv = get_default_conv_template(
                "vicuna"
        ).copy()  # conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        conv.messages = []

        # DETAILS QUESTION
        question = random.choice(FINAL_QUESTIONS).strip()

        s_bbox_string = ''
        num_bboxes = min(len(ori_labels), self.max_gt_per_img)
        for id in range(num_bboxes):
            s_bbox_string = s_bbox_string + f'region{id+1} <bbox>,'
        question = question.replace('<spi_descript>', s_bbox_string)
        conv.append_message(roles["human"], self.begin_str+question)
        answer = self.id_cap_dict[data_item['img_metas'].data['filename'].split('/')[-1]].lower()
        if answer.endswith('.'):
            answer = answer[:-1]
        answer = random.sample(self.sentence_heads, 1)[0] + answer + random.sample(self.sentence_tails, 1)[0]
        conv.append_message(roles["gpt"], answer)
        
        shuffle_ids = torch.randperm(len(ori_labels))
     
        shuffle_ids = shuffle_ids[:self.max_gt_per_img]
        select_bboxes = ori_bboxes[shuffle_ids]
        select_labels = [ori_labels[i] for i in shuffle_ids]


        for i in range(len(select_labels)):
            question = random.choice(REGION_QUESTIONS).strip()
            question = question.replace('<spi_descript>', f'region {i+1}')
            answer = select_labels[i]  # already string
            conv.append_message(roles["human"], question)
            answer = answer.lower()
            if answer.endswith('.'):
                answer = answer[:-1]
            answer = random.sample(self.sentence_heads, 1)[0] + answer + random.sample(self.sentence_tails, 1)[0]
            conv.append_message(roles["gpt"], answer)


        conversations.append(conv.get_prompt())

        image_token_len = (image.shape[1] // 14) * (image.shape[2] // 14)

        for i in range(len(conversations)):
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversations[i] = conversations[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
        


        data_dict = {
            'conversations': conversations
        }


        data_dict['image'] = image

 
        select_bboxes = torch.cat([select_bboxes], dim=0)
        # print('Flickr30k ori_bboxes   >>> ', ori_bboxes)
        select_bboxes = copy.deepcopy(select_bboxes) / image.shape[1]

        data_dict['bboxes'] = select_bboxes
        data_dict['img_metas'] = data_item['img_metas'].data

        return data_dict

    def __getitem__(self, idx):
        data_item = super().__getitem__(idx)
        max_loops = 10
        i = 0

        while True:
            if i > max_loops:
                raise ValueError('No gt_labels')
            i += 1
            if len(data_item['gt_labels']) == 0:
                idx = random.randint(0, len(self)-1)
                data_item = super().__getitem__(idx)
            else:
                break
        data_dict = self.process_text(data_item=data_item)

        return (
            data_dict['conversations'],
            data_dict['image'],
            data_dict['bboxes'],
            data_dict['img_metas']
        )


class VGDATA(CocoDataset):
    CLASSES = ('object',)

    def __init__(self,
                 tokenizer,
                 multimodal_cfg=None,
                 vis_processor=None,
                 ann_file=None,
                 img_prefix=None,
                 add_eos=True,
                 ignore_instruction=True,
                 filter_small=False,
                 test_mode=False,
                 max_gt_per_img=15,
                 ):

        self.multimodal_cfg = multimodal_cfg
        self.tokenizer = tokenizer
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.vis_processor = vis_processor
        self.max_gt_per_img = max_gt_per_img
        self.add_eos = add_eos
        self.ignore_instruction = ignore_instruction
        self.filter_small = filter_small
        self.test_mode = test_mode

        img_norm_cfg = dict(
            mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
            std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
            to_rgb=True)

        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(224, 224), keep_ratio=False),
            # dict(type='RandomShift', shift_ratio=0.5, max_shift_px=32),
            dict(type='FilterAnnotationsFlickr', min_gt_bbox_wh=(2.0, 2.0)),
            dict(type='RandomFlip', flip_ratio=0.),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='DefaultFormatBundleFlickr'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]

        test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(224, 224), keep_ratio=False),
            dict(type='FilterAnnotationsFlickr', min_gt_bbox_wh=(2.0, 2.0)),
            dict(type='RandomFlip', flip_ratio=0.),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=224),
            dict(type='DefaultFormatBundleFlickr'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]

        if test_mode:
            pipeline = test_pipeline
        else:
            pipeline = train_pipeline

        if test_mode:
            ann_file = self.ann_file
            img_prefix = self.img_prefix
        else:
            ann_file = self.ann_file
            img_prefix = self.img_prefix
        train = dict(
            ann_file=ann_file,
            img_prefix=img_prefix,
            test_mode=False,
            pipeline=pipeline, )
        super(CocoDataset, self).__init__(**train)
        # TODO filter the small image? < 32 ?
        self.num_classes = len(self.CLASSES)

        self.begin_str = """The <image> provides an overview of the picture.\n"""
    
        self.sentence_heads = [
            "Sure, there is a ",
            "OK, it is a ",
            "There is a ",
            "It is a ",
        ]
        self.sentence_tails =[
            " in the region.",
            ".",
            " in this region."
        ]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # TODO: obtain images that contain annotation
        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            # convert data type for flickr
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])

            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        # flickr
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                # flickr label
                # TODO
                gt_labels.append(ann['caption'])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def process_text(self, data_item):
        if isinstance(data_item['img'], list):
            # test model
            data_item = {k: v[0] for k, v in data_item.items()}

        return self.train_process_test(data_item)

    def train_process_test(self, data_item):
        image = data_item['img'].data
        ori_labels = data_item['gt_labels']
        ori_bboxes = data_item['gt_bboxes'].data

        conversations = []

        conv = get_default_conv_template(
                "vicuna"
        ).copy()  # conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        conv.messages = []

        shuffle_ids = torch.randperm(len(ori_labels))
        if len(shuffle_ids) > self.max_gt_per_img:
            shuffle_ids = shuffle_ids[:self.max_gt_per_img]
        select_bboxes = ori_bboxes[shuffle_ids]
        select_labels = [ori_labels[i] for i in shuffle_ids]

        for i in range(len(select_labels)):
            question = random.choice(FINAL_QUESTIONS).strip()
            question = question.replace('<spi_descript>', f'region{i+1} <bbox>')
            answer = select_labels[i]  # already string
            if i == 0:
                question = self.begin_str + question
            conv.append_message(roles["human"], question)
            if answer.startswith('a '):
                answer = answer[2:]
            answer = random.sample(self.sentence_heads, 1)[0] + answer + random.sample(self.sentence_tails, 1)[0]
            conv.append_message(roles["gpt"], answer)

        conversations.append(conv.get_prompt())

        image_token_len = (image.shape[1] // 14) * (image.shape[2] // 14)

        for i in range(len(conversations)):
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversations[i] = conversations[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

        data_dict = {
            'conversations': conversations
        }

        data_dict['image'] = image
        # print('vg ori_bboxes   >>> ', ori_bboxes)
        select_bboxes = copy.deepcopy(select_bboxes) / image.shape[1]

        data_dict['bboxes'] = select_bboxes
        data_dict['img_metas'] = data_item['img_metas'].data

        return data_dict

    def __getitem__(self, idx):
        data_item = super().__getitem__(idx)
        max_loops = 10
        i = 0
        while True:
            if i > max_loops:
                raise ValueError('No gt_labels')
            i += 1
            if len(data_item['gt_labels']) == 0:
                idx = random.randint(0, len(self) - 1)
                data_item = super().__getitem__(idx)
            else:
                break
        data_dict = self.process_text(data_item=data_item)

        return (
            data_dict['conversations'],
            data_dict['image'],
            data_dict['bboxes'],
            data_dict['img_metas']
        )


class JackVanillaDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 512
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
        with open(os.path.join(anno_path, 'jack_v20_filter_5000.json')) as f:
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
        '[0,0,0,0]' in bboxes_str[0] or '[]' in bboxes_str[0] or 'white and multi-storied and garage' in bboxes_str[0] \
          or  'lap:[220, 151, 305]' in bboxes_str[0] or 'yellow and blue [equipment]' in bboxes_str[0]:
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

            replace_name = -2
            # random sample convs from start_id -> note logical reasoning NOT has this
            len_conv = len(source)
            start_id = 0
            if len_conv > 2:
                rand_id = random.randint(0, len_conv-1)
                start_id = random.sample([rand_id, int(len_conv//2), int(len_conv//4), int(len_conv//6), int(len_conv//8)], 1)[0]
                start_id = start_id // 2 * 2
                source = source[start_id:]

            
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]
            conv.messages = []


            for j, sentence in enumerate(source):  # note here: the model_max_length only contains about 6-7 VQAs
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{j}"

                if j % 2 == 0:
                    
                    if '<it>' in sentence['value']:
                        sentence['value'] = sentence['value'].replace('<it>', '*it*')
                    if '<region>' in sentence['value']:
                        sentence['value'] = sentence['value'].replace('<region>', '*region*')

                    # 'the cup is on the desk. <cup:[238, 249, 298, 511], red and orange desk:[241, 289, 300, 390]>'
                    # extract the bboxes string: <cup:[238, 249, 298, 511], red and orange desk:[241, 289, 300, 390]>
                    bboxes_str = re.findall(r"<(.+?)>", sentence["value"])
                    # extract the bboxes : [[238, 249, 298, 511], [241, 289, 300, 390]]
                    bboxes = self.strbbox2bbox(bboxes_str)
                    # delete the bboxes string: 'the cup is on the desk.'
                    sentence['value'] = sentence['value'][:sentence['value'].find('<')]

                    if len(bboxes) > 0:
                        if "*it*" in sentence['value']:
                            sentence['value'] = sentence['value'].replace('*it*', 'it <bbox> ')
                        elif "*region*" in sentence['value']:  # region understanding
                            sentence['value'] = sentence['value'].replace('*region*', 'region <bbox> ')
                        else:  # vanilla jack dataset
                            if len(bboxes) == 1:
                                rr_rate = random.random()
                                if rr_rate < 0.3:  # replace the instance name to one of replace_names
                                    ins_name = bboxes_str[0].split('<')[-1].split(':')[0]
                                    if 'the ' + ins_name in sentence['value']:
                                        ins_name = 'the ' + ins_name
                                    replace_name = j
                                    name_re = random.sample(self.replace_names, 1)[0]
                                    sentence['value'] = sentence['value'].replace(ins_name, name_re + ' <bbox> ')


                                    bboxes = [bboxes[0][0], bboxes[0][1], bboxes[0][2], bboxes[0][3]]
                                    w, h = images_ori.shape[:2]

                                    bboxes_human.append(torch.tensor(bboxes) / torch.tensor([h, w, h, w], dtype=torch.half))
                                else:
                                    bboxes = []
                            else:
                                bboxes = []
                                # num_sample = random.sample([0, 1], 1)[0]  #  randomly sample 0 or 1 bbox
                                # bboxes = random.sample(bboxes, num_sample)  # only keep 0/1 region -> click 0/1 box on the image

                    
                    if j == 0:
                        sentence["value"] = '<image>\n' + ' ' + sentence["value"]  # put <image> in the most front

                elif j % 2 == 1:
                    
                    if '<' in sentence['value']:
                        sentence['value'] = sentence['value'][:sentence['value'].find('<')]

                    if replace_name == j - 1:
                        if ins_name in sentence["value"]:
                            sentence["value"] = sentence["value"].replace(ins_name, name_re)

                conv.append_message(role, sentence["value"])

                if len(bboxes_human) > 0 and j % 2 == 1:
                    break

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

            if len(bboxes_human) > 0 and conversations[0].count("<im_start>") == 1:
                bbox = bboxes_human[0]
                break
        
        return (
            conversations,
            images_clip,
            bbox[None, :],
            images,
        )


class JackReferringDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 512
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
        with open(os.path.join(anno_path, 'jack_referring_v10.json')) as f:
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
        # print('bboxes_raw  >> ', bboxes_raw)

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
        '[0,0,0,0]' in bboxes_str[0] or '[]' in bboxes_str[0] or 'white and multi-storied and garage' in bboxes_str[0] \
          or  'lap:[220, 151, 305]' in bboxes_str[0] or 'yellow and blue [equipment]' in bboxes_str[0]:
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
        # print('**********************************************************************')

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
            image_token_len = (images_clip.shape[1] // 14) * (
                    images_clip.shape[2] // 14
            )  # FIXME: 14 is hardcoded patch size

            images, _, ratios = self.transform(Image.fromarray(images_ori))  # preprocess images for dino, check this

            source = self.jack_json[idx]["conversation"]
            
            conv = get_default_conv_template(
                "vicuna"
            ).copy()  # conversation_lib.default_conversation.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

            conversations = []
            bboxes_human = []
            bboxes_gpt = []

            replace_name = -2
            # random sample convs from start_id -> note logical reasoning NOT has this
            len_conv = len(source)
            start_id = 0
            if len_conv > 2:
                rand_id = random.randint(0, len_conv-1)
                start_id = random.sample([rand_id, int(len_conv//2), int(len_conv//4), int(len_conv//6), int(len_conv//8)], 1)[0]
                start_id = start_id // 2 * 2
                source = source[start_id:]

            
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]
            conv.messages = []


            for j, sentence in enumerate(source):  # note here: the model_max_length only contains about 6-7 VQAs
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{j}"

                if j % 2 == 0:
                    
                    if '<it>' in sentence['value']:
                        sentence['value'] = sentence['value'].replace('<it>', '*it*')
                    if '<region>' in sentence['value']:
                        sentence['value'] = sentence['value'].replace('<region>', '*region*')

                    # 'the cup is on the desk. <cup:[238, 249, 298, 511], red and orange desk:[241, 289, 300, 390]>'
                    # extract the bboxes string: <cup:[238, 249, 298, 511], red and orange desk:[241, 289, 300, 390]>
                    bboxes_str = re.findall(r"<(.+?)>", sentence["value"])
                    # extract the bboxes : [[238, 249, 298, 511], [241, 289, 300, 390]]
                    bboxes = self.strbbox2bbox(bboxes_str)
                    # delete the bboxes string: 'the cup is on the desk.'
                    sentence['value'] = sentence['value'][:sentence['value'].find('<')]

                    if len(bboxes) > 0:
                        if "*it*" in sentence['value']:
                            sentence['value'] = sentence['value'].replace('*it*', 'it <bbox> ')
                        elif "*region*" in sentence['value']:  # region understanding
                            sentence['value'] = sentence['value'].replace('*region*', 'region <bbox> ')


                        if len(bboxes) == 1:                            
                            bboxes = [bboxes[0][0], bboxes[0][1], bboxes[0][2], bboxes[0][3]]
                            w, h = images_ori.shape[:2]

                            bboxes_human.append(torch.tensor(bboxes) / torch.tensor([h, w, h, w], dtype=torch.half))
                    
                    if j == 0:
                        sentence["value"] = '<image>\n' + ' ' + sentence["value"]  # put <image> in the most front

                elif j % 2 == 1:
                    
                    if '<' in sentence['value']:
                        sentence['value'] = sentence['value'][:sentence['value'].find('<')]

                    if replace_name == j - 1:
                        if ins_name in sentence["value"]:
                            sentence["value"] = sentence["value"].replace(ins_name, name_re)

                conv.append_message(role, sentence["value"])

                if len(bboxes_human) > 0 and j % 2 == 1:
                    break

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


            # images = self.preprocess(torch.from_numpy(images).permute(2, 0, 1).contiguous())
            images = self.preprocess(torch.from_numpy(np.array(images)).permute(2, 0, 1).contiguous())

            #### postprocess bbox
            bboxes_gpt = self.postprocess_bbox(bboxes_gpt, ratios)  # for DINO prediction

            
            if len(bboxes_human) > 0 and conversations[0].count("<im_start>") == 1:
                # print('len of bboxes > 0  ... ', bboxes_human)
                bbox = bboxes_human[0]
                break
        
        return (
            conversations,
            images_clip,
            bbox[None, :],
            images,
        )


class RefCOCOsDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 512
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
        with open(os.path.join(anno_path, 'jack_refcoco_refcoco+_refcocog_referring_v10.json')) as f:
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

            scaled_boxes = box_xywh_to_cxcywh(scaled_boxes)
            scaled_boxes = scaled_boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            boxes_gt.append(scaled_boxes)
        return boxes_gt

    def strbbox2bbox(self, bboxes_str=None):
        if len(bboxes_str) == 0 or ";" in bboxes_str[0] or 'x1' in bboxes_str[0] or '?' in bboxes_str[0] or \
        '[0,0,0,0]' in bboxes_str[0] or '[]' in bboxes_str[0] or 'white and multi-storied and garage' in bboxes_str[0] \
          or  'lap:[220, 151, 305]' in bboxes_str[0] or 'yellow and blue [equipment]' in bboxes_str[0]:
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
            image_token_len = (images_clip.shape[1] // 14) * (
                    images_clip.shape[2] // 14
            )  # FIXME: 14 is hardcoded patch size

            images, _, ratios = self.transform(Image.fromarray(images_ori))  # preprocess images for dino, check this

            source = self.jack_json[idx]["conversation"]
            
            conv = get_default_conv_template(
                "vicuna"
            ).copy()  # conversation_lib.default_conversation.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

            conversations = []
            bboxes_human = []
            bboxes_gpt = []

            replace_name = -2
            # random sample convs from start_id -> note logical reasoning NOT has this
            len_conv = len(source)
            start_id = 0
            if len_conv > 2:
                rand_id = random.randint(0, len_conv-1)
                start_id = random.sample([rand_id, int(len_conv//2), int(len_conv//4), int(len_conv//6), int(len_conv//8)], 1)[0]
                start_id = start_id // 2 * 2
                source = source[start_id:]

            
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]
            conv.messages = []


            for j, sentence in enumerate(source):  # note here: the model_max_length only contains about 6-7 VQAs
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{j}"

                if j % 2 == 0:
                    
                    if '<it>' in sentence['value']:
                        sentence['value'] = sentence['value'].replace('<it>', '*it*')
                    if '<region>' in sentence['value']:
                        sentence['value'] = sentence['value'].replace('<region>', '*region*')

                    # 'the cup is on the desk. <cup:[238, 249, 298, 511], red and orange desk:[241, 289, 300, 390]>'
                    # extract the bboxes string: <cup:[238, 249, 298, 511], red and orange desk:[241, 289, 300, 390]>
                    bboxes_str = re.findall(r"<(.+?)>", sentence["value"])
                    # extract the bboxes : [[238, 249, 298, 511], [241, 289, 300, 390]]
                    bboxes = self.strbbox2bbox(bboxes_str)
                    # delete the bboxes string: 'the cup is on the desk.'
                    sentence['value'] = sentence['value'][:sentence['value'].find('<')]

                    if len(bboxes) > 0:
                        if "*it*" in sentence['value']:
                            sentence['value'] = sentence['value'].replace('*it*', 'it <bbox> ')
                        elif "*region*" in sentence['value']:  # region understanding
                            sentence['value'] = sentence['value'].replace('*region*', 'region <bbox> ')


                        if len(bboxes) == 1:                            
                            bboxes = [bboxes[0][0], bboxes[0][1], bboxes[0][0]+bboxes[0][2], bboxes[0][1]+bboxes[0][3]]
                            w, h = images_ori.shape[:2]

                            bboxes_human.append(torch.tensor(bboxes) / torch.tensor([h, w, h, w], dtype=torch.half))
                    
                    if j == 0:
                        sentence["value"] = '<image>\n' + ' ' + sentence["value"]  # put <image> in the most front

                elif j % 2 == 1:
                    
                    if '<' in sentence['value']:
                        sentence['value'] = sentence['value'][:sentence['value'].find('<')]

                    if replace_name == j - 1:
                        if ins_name in sentence["value"]:
                            sentence["value"] = sentence["value"].replace(ins_name, name_re)

                conv.append_message(role, sentence["value"])

                if len(bboxes_human) > 0 and j % 2 == 1:
                    break

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

            
            if len(bboxes_human) > 0 and conversations[0].count("<im_start>") == 1:
                # print('len of bboxes > 0  ... ', bboxes_human)
                bbox = bboxes_human[0]
                break

        return (
            conversations,
            images_clip,
            bbox[None, :],
            images,
        )


def collate_fn(batch, tokenizer=None):

    images_list = []
    images_clip_list = []
    conversation_list = []
    bboxes_human_list = []
    offset_list = [0]
    cnt = 0
    # inferences = []
    for (
            conversations,
            image,
            bboxes,
            img_metas,
    ) in batch:

        images_list.append(img_metas)
        images_clip_list.append(image)
        conversation_list.extend(conversations)
        bboxes_human_list.append(bboxes)
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
        'images': torch.stack(images_clip_list.copy(), dim=0),
        'images_clip': torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "bboxes_human_list": bboxes_human_list,
        "offset": torch.LongTensor(offset_list),
        "conversation_list": conversation_list,
    }


class ReferringDataset(torch.utils.data.Dataset):


    def __init__(
            self,
            base_coco17_dir,
            base_coco14_dir,
            base_vg_dir,
            base_flickr_dir,
            tokenizer,
            vision_tower,
            precision: str = "fp32",
            image_size: int = 224,
            dataset="jackvanilla||jackreferring||coco||refcoco||refcocop||refcocog||flick",
            sample_rate=[2, 1, 1, 2, 2, 2],
    ):
        # self.exclude_val = exclude_val
        dataset="jackvanilla||jackreferring||refcocos||coco||refcoco||refcocop||refcocog||vg||flick"
        sample_rate = [2, 3, 3, 1, 1, 1, 1, 1, 1]

        self.dataset = dataset
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")

        image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        multimodal_cfg = dict(
            is_multimodal=True,
            sep_image_conv_front=True,
            image_token_len=256,
            image_aspect_ratio=1,
            use_im_start_end=True,
            image_processor=image_processor)

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "jackvanilla":
                self.all_datasets.append(
                    JackVanillaDataset(
                        base_root='/home/TianYunjie/Workspace/datasets/VG/VG/',
                        vision_tower=vision_tower,
                        tokenizer=tokenizer,
                        anno_path='/home/TianYunjie/Workspace/PycharmProjects/Jack_pure/data_files/'
                    )
                )
            elif dataset == "jackreferring":
                self.all_datasets.append(
                    JackReferringDataset(
                        base_root='/home/TianYunjie/Workspace/datasets/VG/VG/',
                        vision_tower=vision_tower,
                        tokenizer=tokenizer,
                        anno_path='/home/TianYunjie/Workspace/PycharmProjects/Jack_pure/data_files/'
                    )
                )
            elif dataset == "refcocos":
                self.all_datasets.append(
                    RefCOCOsDataset(
                        base_root=base_coco14_dir,
                        vision_tower=vision_tower,
                        tokenizer=tokenizer,
                        anno_path='/home/TianYunjie/Workspace/PycharmProjects/Jack_pure/data_files/'
                    )
                )
            elif dataset == "coco":
                self.all_datasets.append(
                    CocoDet(
                        vis_root=base_coco17_dir,
                        tokenizer=tokenizer,
                        multimodal_cfg=multimodal_cfg,
                    )
                )
            elif dataset == "refcoco":
                self.all_datasets.append(
                    RefCOCO(
                        tokenizer=tokenizer,
                        ann_file='/home/TianYunjie/Workspace/PycharmProjects/Jack_pure/OpenSource/finetune_refcoco_train.json',
                        img_prefix=base_coco14_dir,
                        multimodal_cfg=multimodal_cfg,
                    )
                )
            elif dataset == "refcocop":
                self.all_datasets.append(
                    RefCOCOP(
                        tokenizer=tokenizer,
                        ann_file='/home/TianYunjie/Workspace/PycharmProjects/Jack_pure/OpenSource/finetune_refcoco+_train.json',
                        img_prefix=base_coco14_dir,
                        multimodal_cfg=multimodal_cfg,
                    )
                )
            elif dataset == "refcocog":
                self.all_datasets.append(
                    RefCOCOG(
                        tokenizer=tokenizer,
                        ann_file='/home/TianYunjie/Workspace/PycharmProjects/Jack_pure/OpenSource/finetune_refcocog_train.json',
                        img_prefix=base_coco14_dir,
                        multimodal_cfg=multimodal_cfg,
                    )
                )
            elif dataset == "vg":
                self.all_datasets.append(
                    VGDATA(
                        tokenizer=tokenizer,
                        ann_file='/home/TianYunjie/Workspace/PycharmProjects/Jack_pure/OpenSource/vg_train.json',
                        img_prefix=base_vg_dir,
                        multimodal_cfg=multimodal_cfg,
                    )
                )
            elif dataset == "flick":
                self.all_datasets.append(
                    Flickr30k(
                        tokenizer=tokenizer,
                        ann_file='/home/TianYunjie/Workspace/PycharmProjects/Jack_pure/OpenSource/final_flickr_separateGT_train.json',
                        img_prefix=base_flickr_dir,
                        multimodal_cfg=multimodal_cfg,
                    )
                )


    def __len__(self):
        return 100000

    def __getitem__(self, idx):

        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]

        return data[0]
