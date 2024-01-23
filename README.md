# ChatterBox: Multi-round Multimodal Referring and Grounding


*The paper, code, model, and dataset will be released soon.*
## Documentation

## Getting Started  -->

# <img src="assets/chatterbox_logo_1.png" alt="Alt text for the image" width="45" height="40"> ChatterBox: Multi-round Multimodal Referring and Grounding

 [[Paper](https://arxiv.org/abs/2310.07704)]

[Yunjie Tian*](https://sunsmarterjie.github.io/), [Tianren Ma*](), [Lingxi Xie](), [Jihao Qiu](), [Xi Tang](), [Yuan Zhang](), [Jianbin Jiao](), [Qi Tian](), [Qixiang Ye]() 
[*: equal contribution]


## Overview

<p align="center">
    <img src="assets/figure-structure-1.jpg" width="100%"></a> <br>
    Diagram of ChatterBox Model.
</p>


Key Contributions:

* **MRG** - introduce a new task named **multi-round multimodal referring and grounding** (MRG).
* **CB-300k** - propose a **data construction scheme **and establish the **CB-300K benchmark** to facilitate the research in MRG.
* **Chatterbox Model** - vision-language model that injects explicit **vision modules** into an MLLM, providing an agile and effective solution of MRG.


## Release



## Contents

- [Install](#install)
- [Train](#train)
- [Evaluation](#evaluation)

## Install

1. Clone this repository and navigate to ChatterBox folder

```bash
git clone https://github.com/sunsmarterjie/ChatterBox
cd ChatterBox
```

2. Install Package

```Shell
conda create -n chatterbox python=3.11.5 
conda activate chatterbox
pip install --upgrade pip  # enable PEP 660 support
pip install -r requirements.txt
pip install deepspeed
cd mmcv-1.4.7/
MMCV_WITH_OPS=1 pip install -e .
cd ../model/GroundingDINO/ops
python setup.py build install
```


## Train

ChatterBox 13B is trained on 8 A800 GPUs with 80GB memory. 

```
python startup_chatterbox.py
```


## Evaluation



If this project has been helpful or if you've used our dataset, please cite:
```bash
@inproceedings{tian2024chatterbox,
  title={ChatterBox: Multi-round Multimodal Referring and Grounding},
  author={Tian, Yunjie and Ma, Tianren and Xie, Lingxi and Qiu, Jihao and Tang, Xi and Zhang, Yuan and Jiao, Jianbin and Tian, Qi and Ye, Qixiang},
  booktitle={arxiv xxxx},
  year={2024}
}
```
