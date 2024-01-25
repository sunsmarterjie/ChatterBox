import json
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--coco_path',default="/path/to/MSCOCO2017/annotations/instances_val2017.json")
parser.add_argument('--id_path',default="/path/to/coco_val_id_name.json")
args = parser.parse_args()

def random_category(category,category_one_img):
    category_list=[]
    category_list.append(category)
    diff=list(set(category_one_img)-set(category_list))
    idx=random.randint(0,len(diff)-1)
    return diff[idx]

with open(args.coco_path,'r') as fr1:
    file = json.load(fr1)
    category_list = file['categories']
    category_dict = {}
    for c in category_list:
        id=c['id']
        name=c['name']
        category_dict[id]=name

with open(args.id_path,'r') as f:
    file=json.load(f)
output_list=[]
grouding_list=[]
referring_list=[]
referring_choose_list=[]
for k in file.keys():
    image_id=k
    value=file[k]
    wh=value['wh']
    filename=value["file_name"]
    skip=0
    if 'anno' in value.keys():
        anno=value['anno']
        category_one_img=[]
        for a in anno:
            category=a['category']
            category_one_img.append(category)
            category_one_img2=list(set(category_one_img))
            if len(category_one_img2) < len(category_one_img):
                skip=1
            else:
                skip=0
                category_one_img=category_one_img2
        if skip==0:
            for a in anno:
                category=a['category']
                category_id=a['category_id']
                bbox=a['bbox']#lx,ly,rx,ry
                bbox_g=[bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]]
                question_g='Where is the '+ category+'?'
                answer_g='<'+category+':'+str(bbox_g)+'>'
                question_r='What is it <bbox>?'+str(bbox)
                answer_r='It is a '+category+'.'
                d={'image_id':image_id,'filename':filename,'wh':wh,'catgory':category,'bbox':bbox}
                d_g={'image_id':image_id,'filename':filename,'wh':wh,'catgory':category,'question':question_g,'answer':answer_g,'gt_box':bbox_g}
                d_r={'image_id':image_id,'filename':filename,'wh':wh,'catgory':category,'question':question_r,'answer':answer_r,'bbox':bbox}
                output_list.append(d)
                grouding_list.append(d_g)
                referring_list.append(d_r)
                if len(category_one_img)>1:
                    category_other=random_category(category,category_one_img)
                    b=[]
                    b.append(bbox)
                    bbox=b
                    question_r_choose1 = 'Is it <bbox> a '+ category+' or a '+category_other+'?' + str(bbox)
                    question_r_choose2 = 'Is it <bbox> a '+ category_other+' or a '+category+'?' + str(bbox)
                    if random.random()>0.5:
                        question_r_choose=question_r_choose1
                    else:
                        question_r_choose=question_r_choose2
                    answer_r_choose = 'It is a ' + category + '.'
                    d_r2={'image_id':image_id,'filename':filename,'wh':wh,'catgory':category,'question':question_r_choose,'answer':answer_r,'bbox':bbox}
                    referring_choose_list.append(d_r2)

with open('grouding_qa.json','w') as fw:
    json.dump(grouding_list,fw,indent=1)