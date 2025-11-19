> ## 🚨 Project Discontinued
> 
> This project is no longer maintained. Please visit our new and upgraded project **SuperCCM** 👉 https://github.com/qlnfm/SuperCCM
> 
> ### ❓ What is SuperCCM?
> **SuperCCM** is an open-source Python framework for analyzing corneal nerve images obtained from Corneal Confocal Microscopy (CCM).
> 
> ### 🔗 How is SuperCCM related to this project?
> SuperCCM fully **covers all the functionalities** of this project while being **faster, more accurate, and more feature-rich**. You can think of SuperCCM as the **next-generation upgrade** of this project.
> 
> ### 🛠️ Why create a new SuperCCM project instead of updating this one?
> As we developed more and more components, we wanted a **unified software package** to integrate them, rather than scattering them across multiple repositories. SuperCCM was created to achieve this goal.
> 

# AiCCMetrics

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## About

AiCCMetrics: Next-generation Neural Fiber Morphology Quantification Tool

AiCCMetrics is an open-source project designed to quantification neural fiber morphology. It is licensed under the GNU
General Public License v3.0 (GPL-3.0).

## Features
<hr>

- More efficient
- More accurate
- Open source



##  Cite our paper
```
Qiao Q, Cao J, Xue W, et al. Deep learning-based automated tool for diagnosing diabetic peripheral neuropathy. DIGITAL HEALTH. 2024;10. doi:10.1177/20552076241307573
```


## How to use
<hr>

To use this project, follow these steps:

1. Environmental preparation
```shell
conda create -n qccm python=3.8
conda activate aiccm
pip install -r requirements.txt
```

2. Example
```python
from process.processor import Processor
from utils.calculate import get_CNFL, get_CNFD, get_CNBD
from utils.common import show_image, save_image
from process.draw import draw_result_image


test_image_path = './assets/test.jpg'
segmenter_model_path = './models/nerve.onnx'
result_image_path = './assets/result.png'

p = Processor()
p.set_model_path(segmenter_model_path)
p.load_model()
p.load_image(test_image_path)
p.process()
print(f'CNFL: {get_CNFL(p)}\nCNFD: {get_CNFD(p)}\nCNBD: {get_CNBD(p)}')
```


