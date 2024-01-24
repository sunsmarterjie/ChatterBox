<div align="center">
<h1>ChatterBox </h1>

<h3><img src="assets/chatterbox_logo_1.png" alt="Alt text for the image" width="25" height="25"> ChatterBox: Multi-round Multimodal Referring and Grounding</h3>



[Yunjie Tian*](https://sunsmarterjie.github.io/)<sup>1</sup>, Tianren Ma*<sup>1</sup>, [Lingxi Xie](https://scholar.google.com.hk/citations?user=EEMm7hwAAAAJ&hl=zh-CN&oi=ao)<sup>2</sup>, Jihao Qiu<sup>1</sup>, Xi Tang<sup>1</sup>, Yuan Zhang<sup>1</sup>, Jianbin Jiao<sup>1</sup>, [Qi Tian](https://scholar.google.com.hk/citations?user=61b6eYkAAAAJ&hl=zh-CN&oi=ao)<sup>2</sup>, [Qixiang Ye](https://scholar.google.com.hk/citations?user=tjEfgsEAAAAJ&hl=zh-CN&oi=ao)<sup>1</sup> 

<sup>1</sup>  University of Chinese Academy of Sciences, <sup>2</sup>  HUAWEI Inc.

Paper: ([arXiv 2401.10166](https://arxiv.org/abs/2401.10166))

</div>


## Abstract

In this study, we establish a baseline for a new task named multimodal multi-round referring and grounding (MRG), opening up a promising direction for instance-level multimodal dialogues. We present a new benchmark and an efficient vision-language model for this purpose. The new benchmark, named CB-300K, spans challenges including multi-round dialogue, complex spatial relationships among multiple instances, and consistent reasoning, which are beyond those shown in existing benchmarks. The proposed model, named ChatterBox, utilizes a two-branch architecture to collaboratively handle vision and language tasks. By tokenizing instance regions, the language branch acquires the ability to perceive referential information. Meanwhile, ChatterBox feeds a query embedding in the vision branch to a token receiver for visual grounding. A two-stage optimization strategy is devised, making use of both CB-300K and auxiliary external data to improve the model's stability and capacity for instance-level understanding. Experiments show that ChatterBox outperforms existing models in MRG both quantitatively and qualitatively, paving a new path towards multimodal dialogue scenarios with complicated and precise interactions. 


## Overview

<p align="center">
    <img src="assets/figure-structure-1.jpg" width="80%"></a> <br>
    The architecture of the ChatterBox model. 
</p>


Key Contributions:

* **CB-300k** - propose a **data construction scheme **and establish the **CB-300K benchmark** to facilitate the research in MRG.
* **Chatterbox Model** - vision-language model that injects explicit **vision modules** into an MLLM, providing an agile and effective solution of MRG.


## Updates
 * **` Jan. 24th, 2024`:** The paper, code, model, and dataset will be released soon.* [[Paper](https://arxiv.org/abs/2310.07704)]


## Release



## Contents

- [Install](#install)
- [Train](#train)
- [Evaluation](#evaluation)
- [Demo](#demo)

## Install

1. Clone this repository and navigate to ChatterBox folder

```bash
git clone https://github.com/sunsmarterjie/ChatterBox
cd ChatterBox
```

2. Install Packages

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

ChatterBox is trained on 8 A800 GPUs with 80GB memory. 

```
python startup_chatterbox.py
```



## Evaluation

See details at [evaluation](evaluation/readme.md).


## Demo

xxx.


## Citation

If this project has been helpful or if you've used our dataset, please cite:
```
@article{tian2024chatterbox,
  title={ChatterBox: Multi-round Multimodal Referring and Grounding},
  author={Tian, Yunjie and Ma, Tianren and Xie, Lingxi and Qiu, Jihao and Tang, Xi and Zhang, Yuan and Jiao, Jianbin and Tian, Qi and Ye, Qixiang},
  journal={arXiv preprint arXiv:2401.10166},
  year={2024}
}
```

## Acknowledgment

This project is based on LLaVA ([paper](https://arxiv.org/abs/2304.08485), [code](https://github.com/haotian-liu/LLaVA)), LISA ([paper](https://arxiv.org/abs/2308.00692), [code](https://github.com/dvlab-research/LISA)), GPT4RoI ([paper](https://arxiv.org/abs/2307.03601), [code](https://github.com/jshilong/GPT4RoI)), thanks for their excellent works.
