import json
import os.path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--coco_path',default="/path/to/MSCOCO2017/annotations/instances_val2017.json")
args = parser.parse_args()

with open(args.coco_path,'r') as f:
    file=json.load(f)
    image_list=file['images']
    anno_list=file['annotations']
    category_list=file['categories']
    image_dict={}
    category_dict={}
    for c in category_list:
        id=c['id']
        name=c['name']
        category_dict[id]=name
    for i in image_list:
        file_name=i['file_name']
        wh=[i['width'],i['height']]
        id=i['id']
        image_dict[id]={'file_name':file_name,'wh':wh}
    for j in anno_list:
        id=j['image_id']
        bbox_or=j['bbox']
        bbox=[bbox_or[0],bbox_or[1],bbox_or[0]+bbox_or[2],bbox_or[1]+bbox_or[3]]#lx,ly,rx,ry
        category_id=j['category_id']
        category=category_dict[category_id]
        if 'anno' not in image_dict[id].keys():
            image_dict[id]['anno'] = [{'category_id':category_id,'category':category,'bbox':bbox}]
        else:
            image_dict[id]['anno'].append({'category_id':category_id,'category':category,'bbox':bbox})

save_path=os.path.join(os.path.abspath('.'),'coco_val_id_name.json')
with open(save_path,'w') as fw:
    json.dump(image_dict,fw,indent=1)
