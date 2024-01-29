# Grounding
## Results
- Results of single-round visual grounding on MSCOCO2017 test set.


| Model                                                        | mIoU  | Succ. Rate | mIoU @ Succ. |
| ------------------------------------------------------------ | ----- | ---------- | ------------ |
| Chatterbox  - dino                                                 | 0.710 | 0.762      | 0.904        |
| Chatterbox  - groundingdino                                               | 0.678 | 0.718      | 0.914       |
 - _note: mIoU (mean IoU of all cases), Succ. Rate (IoU is at least 0.5), mIoU @ Succ: mean IoU of successful cases._

## Data preparation
### 1. Download image and annotations
- Download the official MSCOCO2017 image dataset from [cocodataset.org](https://cocodataset.org/#download) or [Baidu Netdisk](https://blog.csdn.net/qq_47233366/article/details/126575414)
### 2. Generate questions for grounding

- You can run the following command ([generate_coco_id.py](https://github.com/sunsmarterjie/ChatterBox/blob/main/evaluation/grounding/grounding_dino/generate_coco_id.py), [generate_grounding_qa.py](https://github.com/sunsmarterjie/ChatterBox/blob/main/evaluation/grounding/grounding_dino/generate_grounding_qa.py)) to generate questions and corresponding GT answers:
```python
python generate_coco_id.py --coco_path /path/to/MSCOCO2017/annotations/instances_val2017.json
python generate_grounding_qa.py --coco_path /path/to/MSCOCO2017/annotations/instances_val2017.json --id_path /path/to/coco_val_id_name.json
```
 - Alternatively, you can also download the pre-processed file: [grounding_qa.json](https://github.com/sunsmarterjie/ChatterBox/blob/main/evaluation/grounding/grounding_dino/grounding_files/grounding_qa.json). Remember to replace image and annotation paths in our provided file with the specific path on your machine.

## Evaluation
### 1. Grounding results
#### 1. Chatterbox - groundingdino
- You can run the following command ([eval_grounding.sh](https://github.com/sunsmarterjie/ChatterBox/blob/main/evaluation/grounding/grounding_dino/eval_grounding.sh)) to get predicted results:
```
bash eval_grounding.sh
```
- Remember to replace the related paths in our provided file with the specific path on your machine. Here is the provided [checkpoint](https://pan.baidu.com/s/1eXQdkXVqNCp4Sa31S-rJPw&pwd=7c07).
- Alternatively, you can also download the predicted file: [prediction.json](https://github.com/sunsmarterjie/ChatterBox/blob/main/evaluation/grounding/grounding_dino/grounding_files/prediction.json). Remember to replace image and annotation paths in our provided file with the specific path on your machine.
### 2. Evaluation
- You can run the following command ([evaluate_coco_gd.py](https://github.com/sunsmarterjie/ChatterBox/blob/main/evaluation/grounding/grounding_dino/evaluate_coco_gd.py)) to evaluate results of chatterbox:
```
python evaluate_coco_gd.py --gt_predict_file /path/to/predict_grounding.json
```
