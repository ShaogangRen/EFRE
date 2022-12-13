# EFRE 
Causal Discovery

This is the code for the paper  'Flow-based Perturbation for Cause-effect Inference' (https://dl.acm.org/doi/abs/10.1145/3511808.3557326).

## Requirements
Python 3.8

Causal Discovery Toolbox (CDT)
(https://fentechsolutions.github.io/CausalDiscoveryToolbox/html/index.html).

## Run the Code

```
python GridS_EFRE.py
```
## Change the Setup
You can modify GridS_EFRE.py to reset the dataset, learning rate, etc. You also can use it to do
hyperparameter search.

For the Tuebingen dataset (in CDT Toolbox), the setup in GridS_main.py can 
achieve above 0.92 accuracy. 


Some of the code files were adapted from ffjord (https://github.com/rtqichen/ffjord).

## Citation
```
@inproceedings{ren2022efre,
  title={Flow-based Perturbation for Cause-effect Inference},
  author={Ren, Shaogang and Li, Ping},
  booktitle = {Proceedings of the 31st {ACM} International Conference on Information
               and Knowledge Management (CIKM)}, 
  address   = {Atlanta, GA},
  year      = {2022}
}
```