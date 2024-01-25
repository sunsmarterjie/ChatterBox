import json
import os
import random
from chat import get_prompt, chat
import os.path as op
from polish import polish, get_polish_prompt
from txt2json import txt2json


def get_last_idx(dir):
    filenames=os.listdir(dir)
    filenames.sort(key=lambda x: int(x[:-5]))
    if filenames ==[]:
        last_id=-1
    else:
        last_file=filenames[-1].split('.')[0]
        last_id=int(last_file)
    return last_id,len(filenames)

def get_next_id(dir,last_id):
    filenames = os.listdir(dir)
    filenames.sort(key=lambda x: int(x[:-5]))
    last_name=str(last_id)+'.json'
    if last_id == -1:
        return filenames[0]
    for cnt,f in enumerate(filenames):
        if last_name ==f:
            cnt_next=cnt+1
            return filenames[cnt_next]
    return -1


def shorten(x,n=2):
    x2={}
    for k in x.keys():
        v = x[k]
        v2 = []
        for i in v:
            v2.append(round(i, n))
        x2[k]=v2
    return x2

def read_saveQA(anno_dir,save_dir,modified_response_save_dir,image_info,n_relationship=20,n_question=10):
    last_idx,len_save_dir=get_last_idx(save_dir)
    anno_name=get_next_id(anno_dir,last_idx)
    anno_file=os.path.join(anno_dir,anno_name)
    image_id = anno_name[:-5]
    image_info_2 = image_info[str(image_id)]
    if os.path.exists(anno_file):
        with open(anno_file,'r') as f:
            relationship_list=json.load(f)
            if len(relationship_list)>15:
                n_question=11
            elif len(relationship_list)<=15 and len(relationship_list)>=8:
                n_question = 7
            else:
                n_question=max(len(relationship_list)-1,1)
            if len(relationship_list)>3:
                prompt=get_prompt(relationship_list,n_question)
                response=chat(prompt)

                # optimal: use gpt-3.5 to polish  response from gpt-4
                # prompt2 = get_polish_prompt(response)
                # response_m = polish(prompt2)

                conversation=txt2json(response)
                d={'id':image_id,'image':image_info_2["relative_path"],'image_wh':image_info_2['wh'],'conversation':conversation}
                with open(os.path.join(save_dir,image_id+'.json'),'w') as fw:
                    print(os.path.join(save_dir,image_id+'.json'))
                    json.dump(d,fw,indent=1)
            else:
                d = {'id': image_id, 'image': image_info_2["relative_path"], 'image_wh': image_info_2['wh'],
                     'conversation': []}
                with open(os.path.join(save_dir, image_id + '.json'), 'w') as fw:
                    print(os.path.join(save_dir, image_id + '.json'))
                    json.dump(d, fw, indent=1)

                # optimal: use gpt-3.5 to polish  response from gpt-4
                # with open(os.path.join(modified_response_save_dir,image_id+'.txt'),'w') as fw:
                #     print(os.path.join(modified_response_save_dir,image_id+'.txt'))
                #     fw.write(response_m)
    else:
        d= {'id':image_id,'image':image_info_2["relative_path"],'image_wh':image_info_2['wh'],'conversation':[]}
        with open(os.path.join(save_dir, image_id + '.json'), 'w') as fw:
            print(os.path.join(save_dir, image_id + '.json'))
            # fw.write(response)
            json.dump(d, fw,indent=1)


def get_save_dir(anno_base_dir,save_base_dir):
    anno_dirs=os.listdir(anno_base_dir)# 0 1 2 3
    save_dirs=os.listdir(save_base_dir)# 0 1 2 3
    ls=len(save_dirs)
    lad=len(anno_dirs)
    if ls>0 and ls<=lad:
        last_save_dir=save_dirs[ls-1]
        last_save_dir_abs=op.join(save_base_dir,last_save_dir)
        current_file=os.listdir(last_save_dir_abs)
        lc=len(current_file)
        anno_dir=op.join(anno_base_dir,last_save_dir)
        anno_file=os.listdir(anno_dir)
        la=len(anno_file)
        if lc<la:
            current_save_path=last_save_dir_abs
            current_anno_path=anno_dir
            return current_save_path,current_anno_path
        elif lc == la:
            last_save_dir=save_dirs[ls]
            current_save_path = op.join(save_base_dir, last_save_dir)
            current_anno_path=op.join(anno_base_dir, last_save_dir)
            return current_save_path,current_anno_path
    elif ls==0:
        current_dir=anno_dirs[0]
        current_save_path = op.join(save_base_dir, current_dir)
        current_anno_path = op.join(anno_base_dir,current_dir)
        return current_save_path,current_anno_path
    else:
        return -1

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--base_dir',default='/path/to/base_dir')
args = parser.parse_args()

base_dir=args.base_dir
anno_base_dir=op.join(base_dir, 'relation')
save_base_dir=op.join(base_dir, 'response')
if not op.exists(save_base_dir):
    os.makedirs(save_base_dir)
idx='0'
current_save_path=op.join(save_base_dir,idx)
current_anno_path=op.join(anno_base_dir,idx)
if not op.exists(current_save_path):
    os.makedirs(current_save_path)
modified_response_save_dir=op.join(base_dir, 'm_response',idx)
# optimal: use gpt-3.5 to polish  response from gpt-4
# if not op.exists(modified_response_save_dir):
#     os.makedirs(modified_response_save_dir)
with open('./image_info/exist_image_info.json','r') as f:
    image_info=json.load(f)
read_saveQA(current_anno_path,current_save_path,modified_response_save_dir,image_info)