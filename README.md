# How I Warped Your Noise (ICLR 2024) - Unofficial

An unofficial implementation of the ICLR 2024 paper "How I Warped Your Noise: a Temporally-Correlated Noise Prior for Diffusion Models", Chang et al.

## Introduction

- The Jupyter Notebook file "How_I_Warped_Your_Noise_Unofficial" has been originally written for Google Colab. For direct usage at Colab, we provide the direct [link](https://colab.research.google.com/drive/1_vdMUvRI2sm9q60FAwYaThr1jgJEj2Ie?usp=sharing) to the Colab.

- For local usage, run it on a Conda environment that has einops, ninja, and Nvdiffrast installed.

## Additional Details

Cell 9 contains the following code:
```
  warp_idxs = torch.stack((warp_i, warp_j), dim=-1)
  tgt_to_src_map = warp_idxs
```
Variable `tgt_to_src_map` designates the corresponding locations that each vertex of partitioned polygons of the target frame is mapped to at the source frame (warped locations). 
- The current code provides simple warping configurations, identity mapping (no change), and rotation mapping (slight rotation), for testing purposes.
- Replace the variable with your desired correspondence mapping for your personal usage.

--------

This implementation contains two separate implementations for triangle rasterization, at Cells 12 and 13. 
- Using the code at Cell 12, which uses scatter `torch.scatter_add_()` is more efficient and therefore recommended.
- The code at Cell 13 uses `for` loop for sequential rasterization, and is therefore much slower, but computationally cheaper. 



## Citation
```
@inproceedings{
chang2024how,
title={How I Warped Your Noise: a Temporally-Correlated Noise Prior for Diffusion Models},
author={Pascal Chang and Jingwei Tang and Markus Gross and Vinicius C. Azevedo},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=pzElnMrgSD}
}
```
