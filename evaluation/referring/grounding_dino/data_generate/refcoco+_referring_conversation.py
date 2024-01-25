import torch
import re
import os
import json
import pickle
import random

refs_data_path = "/Workspace/TianYunjie/PycharmProjects/new_jack/refcoco+/refs(unc).p"
json_data_path = "/Workspace/TianYunjie/PycharmProjects/new_jack/refcoco+/instances.json"

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
        ref_id=data['ref_id']
        bbox = anno_dict[data['ann_id']]['bbox']
        bbox=[[int(bbox[0]),int(bbox[1]),int(bbox[2]+bbox[0]),int(bbox[3]+bbox[1])]]
        bbox_str = str(bbox)
        # bbox_str = bbox_str.strip('[')
        # bbox_str = bbox_str.rstrip(']')
        question = 'What is it <bbox>?'+bbox_str
        sentence = data['sentences']
        sentence2=[]
        for cnt,s in enumerate(sentence):
            phrase=s['sent']
            phrase=phrase.lower()
            phrase+='.'
            sentence2.append(phrase)
        answer = sentence2
        data2={'file_name':file_name,'image_id':image_id,'category_id':category,'question':question,'answer':answer,'ref_id':ref_id}
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

with open("/Workspace/TianYunjie/PycharmProjects/new_jack/refcoco+/val2.json", 'w') as fw:
    json.dump(val_list,fw,indent=1)
with open("/Workspace/TianYunjie/PycharmProjects/new_jack/refcoco+/test_a2.json", 'w') as fw:
    json.dump(testa_list,fw,indent=1)
with open("/Workspace/TianYunjie/PycharmProjects/new_jack/refcoco+/test_b2.json", 'w') as fw:
    json.dump(testb_list,fw,indent=1)
with open("/Workspace/TianYunjie/PycharmProjects/new_jack/refcoco+/train2.json", 'w') as fw:
    json.dump(train_list,fw,indent=1)
