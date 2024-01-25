
# Referring
## Results
- Results of single-round visual Referring on Refcocog test set.

| Model                                                        | METEOR  | CIDER | Bert_score |
| ------------------------------------------------------------ | ----- | ---------- | ------------ |
| Chatterbox                                                   | 16.6 | 65.2     |    88.83     |
| Chatterbox(cleaned results)                                  | 16.0 | 71.2      |     88.89     | 

- Results of single-round visual Referring on Refcocog val set.

| Model                                                        | METEOR  | CIDER | Bert_score |
| ------------------------------------------------------------ | ----- | ---------- | ------------ |
| Chatterbox                                                   | 16.7 | 62.2     |     88.54    |
| Chatterbox(cleaned results)                                  | 16.2 | 68.5     |    88.55     |

## Data preparation
### 1. Download image and annotations
- Download the official MSCOCO2014 image dataset from [cocodataset.org](https://cocodataset.org/#download) or [Baidu Netdisk](https://pan.baidu.com/s/1mz3_9IAYD0X8OD8f37ikvQ )(password:p41k)
- Download the annotations for Refcoco/Refcoco+/Refcocog from [Github](https://github.com/lichengunc/refer)

### 2. Generate questions for referring
- You can run the following command [refcocog_referring_conversation.py](https://github.com/sunsmarterjie/ChatterBox/blob/main/evaluation/referring/grounding_dino/data_generate/refcocog_referring_conversation.py) to generate questions and corresponding GT answers for Refcocog referring, we also provide codes for Refcoco and Refcoco+:
```python
python refcocog_referring_conversation.py 
python refcoco_referring_conversation.py
python refcoco+_referring_conversation.py
```
 - Remember to replace annotation paths in our provided file with the specific path on your machine. (refs_data_path, json_data_path)
 - Alternatively, you can also download the pre-processed file from [Baidu Netdisk](https://pan.baidu.com/s/1v6yex_M6Z4bjyNxzYwWQ9w?pwd=gt00)(password:gt00).

## Evaluation
### 1. Referring results
- You can run the following command [refcocog_referring_test.py](https://github.com/sunsmarterjie/ChatterBox/blob/main/refcocog_referring_test.py) to get predicted results:
```
python refcocog_referring_test.py
```
- Remember to replace the related paths in our provided file with the specific path on your machine.(coco2014_path,refcocog_test_path,save_out_path)
- Alternatively, you can also download the predicted file from [raw_output](https://github.com/sunsmarterjie/ChatterBox/tree/main/evaluation/referring/grounding_dino/output/raw_output). 

### 2. Evaluation
#### Data cleaning
- To fit the form of evalution programs, you need to run [process.py](https://github.com/sunsmarterjie/ChatterBox/blob/main/evaluation/referring/grounding_dino/output/process.py)
- Our model will output some words like "it is" or "there is" which may not contained in the GT. You can run the [process_clean.py](https://github.com/sunsmarterjie/ChatterBox/blob/main/evaluation/referring/grounding_dino/output/process_clean.py) to delete because these words don't influence the meaning of the sentence.
- Alternatively, you can also download the processed results from [modify_output](https://github.com/sunsmarterjie/ChatterBox/tree/main/evaluation/referring/grounding_dino/output/modify_output) and [clean_output](https://github.com/sunsmarterjie/ChatterBox/tree/main/evaluation/referring/grounding_dino/output/clean_output). 
#### Offical evaluation
- We evaluate the results of Refcocog with the project [refer](https://github.com/lichengunc/refer), what you need to do is put the [test.py](https://github.com/sunsmarterjie/ChatterBox/blob/main/evaluation/referring/grounding_dino/utils/test.py) under the project's content and run it. The config of the project is different from ours, you may need another enviroment to run it. 
-  Remember to replace image and annotation paths in our provided file with the specific path on your machine. (sample_expr_file)
```
python test.py
```
#### Bert_score evalution
- We also use [bert_score](https://github.com/Tiiiger/bert_score) to evaluate the results
- You need put [bert_score.py](https://github.com/sunsmarterjie/ChatterBox/blob/main/evaluation/referring/grounding_dino/utils/bert_score.py) under the content of [bert_score](https://github.com/Tiiiger/bert_score) to evaluate.
- Remember to replace the related paths in our provided file with the specific path on your machine. (path1,path2)
