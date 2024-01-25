import json
import os.path

# get exist image
base_path='/path/to/Visual Genome/'
relative_path_1='VG_100K'
image_dir_1=os.path.join(base_path,relative_path_1)
image_list_1=os.listdir(image_dir_1)
relative_path_2='VG_100K_2'
image_dir_2=os.path.join(base_path,relative_path_2)
image_list_2=os.listdir(image_dir_2)
exist_image_info={}
exist_image_info_2=[]
exist_image_list=[]
nonexistent_image_info={}
nonexistent_image_info_2={}
nonexistent_image_list=[]
with open("/path/to/Visual Genome/image_data.json",'r') as f:
    file=json.load(f)
    for fi in file:
        image_id=fi['image_id']
        image_url=fi['url']
        image_url_split=image_url.split('/')
        if image_url_split[-1].split('.')[0]==str(image_id):
            wh=[fi['width'],fi['height']]
            #assert image_url_split[-1].split('.')[0]==str(image_id)
            relative_path=image_url_split[-2]
            image_name=image_url_split[-1]
            #abs_path=os.path.join(base_path,relative_path,image_name)
            if relative_path==relative_path_1:
                if image_name in image_list_1:
                    rel_path=relative_path
                elif image_name in image_list_2:
                    rel_path = relative_path_2
                else:
                    nonexistent_image_list.append(image_id)
                    continue
            elif relative_path==relative_path_2:
                if image_name in image_list_2:
                    rel_path=relative_path
                elif image_name in image_list_1:
                    rel_path = relative_path_1
                else:
                    nonexistent_image_list.append(image_id)
                    continue
            rel_path_image=os.path.join(rel_path,image_name)
            d={'image_id':image_id,'relative_path':rel_path_image,'wh':wh}
            exist_image_info_2.append(d)
            exist_image_info[image_id]={'relative_path':rel_path_image,'wh':wh}
            exist_image_list.append(image_id)
        else:
            nonexistent_image_list.append(image_id)
    pass

assert len(nonexistent_image_list)+len(exist_image_list)==108077
exist_image_list_10000=exist_image_list[:10000]
base_dir=os.path.abspath('.')
with open(os.path.join(base_dir,'exist_image_list.json'),'w') as fw:
    json.dump(exist_image_list,fw)
with open(os.path.join(base_dir,'exist_image_info.json'),'w') as fw:
    json.dump(exist_image_info,fw)
with open(os.path.join(base_dir,'nonexistent_image_list.json'),'w') as fw:
    json.dump(nonexistent_image_list,fw)
