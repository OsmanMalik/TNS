# Sampling-Based Decomposition Algorithms for Arbitrary Tensor Networks
This repo contains code used in the experiment in the paper
> O. A. Malik, V. Bharadwaj, R. Murray. 
> *Sampling-Based Decomposition Algorithms for Arbitrary Tensor Networks.*
> Preprint [arXiv:2210.03828](https://arxiv.org/abs/2210.03828), 2022.

A preprint is available [on arXiv](https://arxiv.org/abs/2210.03828).

## Referencing this code

If you use this code in any of your own work, please reference our paper:

```
@misc{malik2022TNS,
  doi = {10.48550/ARXIV.2210.03828},
  url = {https://arxiv.org/abs/2210.03828},
  author = {Malik, Osman Asif and Bharadwaj, Vivek and Murray, Riley},
  title = {Sampling-Based Decomposition Algorithms for Arbitrary Tensor Networks},
  publisher = {arXiv},
  year = {2022},
}
```

## Some further details on the code

- `experiment_TNS.m`: 
This script runs the feature extraction experiment in our paper. 
It is an adaption of the script `experiment_3.m` in [this repo](https://github.com/OsmanMalik/TD-ALS-ES).
- `draw_samples_TNS_CP.m`:
This function draws the samples in the tensor network (TN) matrices that arise in CP decomposition.
It is an adaption of the function `draw_samples_CP.m` in [this repo](https://github.com/OsmanMalik/TD-ALS-ES).
- `TNS_CP.m`:
This function computes the CP decomposition using the TN sampling approach we propose in the paper.
It leverages the function `draw_samples_TNS_CP.m` above to do the actual sampling of the least squares problems.
It is an adaption of the script `cp_als_es.m` in [this repo](https://github.com/OsmanMalik/TD-ALS-ES).
- `draw_samples_TNS_TR.m`:
This function draws the samples in the TN matrices that arise in tensor ring decomposition.
It is an adaption of the function `draw_samples_TR.m` in [this repo](https://github.com/OsmanMalik/TD-ALS-ES).
- `TNS_TR.m`:
This function computes the tensor ring decomposition using the TN sampling approach we propose in the paper.
It leverages the function `draw_samples_TNS_TR.m` above to do the actual sampling of the least squares problems.
It is an adaption of the script `tr_als_es.m` in [this repo](https://github.com/OsmanMalik/TD-ALS-ES).


## Requirements

The code in this repo is reliant on code from the following two repositories:
- https://github.com/OsmanMalik/tr-als-sampled
- https://github.com/OsmanMalik/TD-ALS-ES

Before attempting to run the code in this repo, please ensure that both of the repos listed above are downloaded and made available in Matlab, e.g., via the `addpath` command.
Also, ensure that all dependencies and requirements for those to repos are installed properly.

## Author contact information

Please feel free to contact me at any time if you have any questions or would like to provide feedback on this code or on the paper. 
I can be reached at `oamalik (at) lbl (dot) gov`. 
