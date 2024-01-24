<div align="center">
<h1>ChatterBox </h1>

<h3><img src="assets/chatterbox_logo_1.png" alt="Alt text for the image" width="25" height="25"> ChatterBox: Multi-round Multimodal Referring and Grounding</h3>



[Yunjie Tian*](https://sunsmarterjie.github.io/)<sup>1</sup>, Tianren Ma*<sup>1</sup>, [Lingxi Xie](https://scholar.google.com.hk/citations?user=EEMm7hwAAAAJ&hl=zh-CN&oi=ao)<sup>2</sup>, Jihao Qiu<sup>1</sup>, Xi Tang<sup>1</sup>, Yuan Zhang<sup>1</sup>, Jianbin Jiao<sup>1</sup>, [Qi Tian](https://scholar.google.com.hk/citations?user=61b6eYkAAAAJ&hl=zh-CN&oi=ao)<sup>2</sup>, [Qixiang Ye](https://scholar.google.com.hk/citations?user=tjEfgsEAAAAJ&hl=zh-CN&oi=ao)<sup>1</sup> 

<sup>1</sup>  University of Chinese Academy of Sciences, <sup>2</sup>  HUAWEI Inc.

Paper: ([arXiv 2401.10166](https://arxiv.org/abs/2401.10166))

</div>


## Overview

<p align="center">
    <img src="assets/figure-structure-1.jpg" width="100%"></a> <br>
    Diagram of ChatterBox Model.
</p>


Key Contributions:

* **MRG** - introduce a new task named **multi-round multimodal referring and grounding** (MRG).
* **CB-300k** - propose a **data construction scheme **and establish the **CB-300K benchmark** to facilitate the research in MRG.
* **Chatterbox Model** - vision-language model that injects explicit **vision modules** into an MLLM, providing an agile and effective solution of MRG.


## Updates
 * **` Jan. 19th, 2024`:** The paper, code, model, and dataset will be released soon.* [[Paper](https://arxiv.org/abs/2310.07704)]


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
