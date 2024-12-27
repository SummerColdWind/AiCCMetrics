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

## Attention please
<hr>

> This warehouse only provides the algorithm source code, if you need to use, please build your own. We also offer an online bulk analysis website, please visit us: www.aiccm.fun


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


