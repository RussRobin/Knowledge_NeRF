# Knowledge_NeRF
This is the official implementation for Knowledge NeRF: Few-shot Novel View Synthesis for Dynamic Articulated Objects (Journal of Visual Communication and Image Representation, 2025).

[[JVCI](https://www.sciencedirect.com/science/article/pii/S1047320325002007)]
[[arXiv](http://arxiv.org/abs/2404.00674)]

## Install

Please refer to [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch).

## Datasets

We propose a new dynamic dataset consisting of synthetic and real-world articulated objects, consisting of 2 spaces: original space and deformed space. If you are interested in our dataset, please reach out to `russ000robin@gmail.com`.

[NeRF Synthetic Dataset](https://arxiv.org/abs/2003.08934) and
[Shiny Blender Dataset](https://arxiv.org/abs/2112.03907) 
are also adopted in our paper.

## Run

Our method includes 3 steps:
1. Initialize projection module
2. Train projection module
3. Finetune projection module and original NeRF

Please `bash pipeline.sh` to run Knowledge NeRF.

## Citation
For any questions, please feel free to reach out to start an issue in this repo.

```
@article{CAI2025104586,
title = {Knowledge NeRF: Few-shot novel view synthesis for dynamic articulated objects},
journal = {Journal of Visual Communication and Image Representation},
volume = {112},
pages = {104586},
year = {2025},
issn = {1047-3203},
doi = {https://doi.org/10.1016/j.jvcir.2025.104586},
url = {https://www.sciencedirect.com/science/article/pii/S1047320325002007},
author = {Wenxiao Cai and Xinyue Lei and Xinyu He and Junming Leo Chen and Yuzhi Hao and Yangang Wang},
```
