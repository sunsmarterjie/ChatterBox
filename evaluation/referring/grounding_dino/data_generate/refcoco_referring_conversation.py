import torch
import re
import os
import json
import pickle
import random

refs_data_path = '/home/TianYunjie/Workspace/PycharmProjects/Jack/refcoco/refs(unc).p'
json_data_path = '/home/TianYunjie/Workspace/PycharmProjects/Jack/refcoco/instances.json'

val_dict= {}
testa_dict= {}
testb_dict= {}
image_dict= {}
anno_dict= {}
val_list=[]
testa_list=[]
testb_list=[]
train_list=[]
with open(json_data_path) as f:
    json_data = json.load(f)
    images=json_data['images']
    for image in images:
        image_dict[image['id']]=image
    for anno in json_data['annotations']:
        anno_dict[anno['id']]=anno

with open(refs_data_path, 'rb') as f:
    refs_data = pickle.load(f)
    for data in refs_data:
        split_id=data['split']
        file_name=data['file_name']
        image_id=data['image_id']
        category=data['category_id']
        bbox = anno_dict[data['ann_id']]['bbox']
        bbox=[[int(bbox[0]),int(bbox[1]),int(bbox[2]+bbox[0]),int(bbox[3]+bbox[1])]]
        bbox_str = str(bbox)
        # bbox_str = bbox_str.strip('[')
        # bbox_str = bbox_str.rstrip(']')
        question = 'What is in this region <bbox>?'+bbox_str
        phrase = data['sentences'][0]['sent']
        phrase2=phrase.split(' ')
        if phrase2[0]== 'a' or phrase2[0]== 'the' or phrase2[0]== 'this' or phrase2[0]== 'that':
            p2=''
            for p in range(1,len(phrase2)):
                p2+=phrase2[p]
                p2+=' '
            answer = 'It is ' + p2 + '.'
        else:
            answer = 'It is a ' + phrase + '.'
        data2={'file_name':file_name,'image_id':image_id,'category_id':category,'question':question,'answer':answer}
        if split_id=='val':
            val_dict[data['image_id']]= data
            val_list.append(data2)
        elif split_id == 'testA':
            testa_dict[data['image_id']]= data
            testa_list.append(data2)
        elif split_id == 'testB':
            testb_dict[data['image_id']]= data
            testb_list.append(data2)
        elif split_id == 'train':
            train_list.append(data2)
        else:
            pass

with open('/home/TianYunjie/Workspace/PycharmProjects/Jack/refcoco_refering/val_[[]].json', 'w') as fw:
    json.dump(val_list,fw,indent=1)
with open('/home/TianYunjie/Workspace/PycharmProjects/Jack/refcoco_refering/testa_[[]].json', 'w') as fw:
    json.dump(testa_list,fw,indent=1)
with open('/home/TianYunjie/Workspace/PycharmProjects/Jack/refcoco_refering/testb_[[]].json', 'w') as fw:
    json.dump(testb_list,fw,indent=1)
with open('/home/TianYunjie/Workspace/PycharmProjects/Jack/refcoco_refering/train_[[]].json', 'w') as fw:
    json.dump(train_list,fw,indent=1)

# with open('/refcoco_refering/val.json', 'w') as fw:
#     json.dump(val_list,fw,indent=1)
# with open('/refcoco_refering/testa.json', 'w') as fw:
#     json.dump(testa_list,fw,indent=1)
# with open('/refcoco_refering/testb.json', 'w') as fw:
#     json.dump(testb_list,fw,indent=1)
# with open('/refcoco_refering/train.json', 'w') as fw:
#     json.dump(train_list,fw,indent=1)