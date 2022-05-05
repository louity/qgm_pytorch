# qgm_pytorch

PyTorch implementation of multi-layer quasi-geostrophic model on rectangular domain with solid boundaries, with parameterizatition described in the paper [https://arxiv.org/abs/2204.13914](Modified hyper-viscosity for coarse-resolution ocean models).

## Requirements

```
torch==1.10
numpy==1.20
```
Tested with Intel CPUs and NVIDIA RTX 2080Ti GPU.

## Usage

An example is included in `QGM.py` and can be ran with
```
python QGM.py
```
Model parameters are defined in a python dictionary as follows
```
param = {
    'nx': 97, # number of points in x
    'ny': 121, # number of points in y
    'Lx': 3840.0e3, # Length in the x direction (m)
    'Ly': 4800.0e3, # Length in the y direction (m)
    'nl': 3, # number of layers
    'heights': [350., 750., 2900.], # heights between layers (m)
    'reduced_gravities': [0.025, 0.0125], # reduced gravity numbers (m/s^2)
    'f0': 9.375e-5, # coriolis (s^-1)
    'a_2': 0., # laplacian diffusion coef (m^2/s)
    'a_4': 5.0e11, # bi-laplacian diffusion coef (m^4/s)
    'beta': 1.754e-11, # coriolis gradient (m^-1 s^-1)
    'delta_ek': 2.0, # eckman height (m)
    'dt': 1200., # Time step
    'bcco': 0.2, # boundary condition coef. (non-dim.)
    'tau0': 2.0e-5, # wind stress magnitude m/s^2
    'n_ens': 0, # 0 for no ensemble,
    'device': 'cpu', # 'cuda' for NVIDIA GPUS, otherwise 'cpu'
    'p_prime': '', # parameter for the proposed parameterization
}
```
If you use this code, please cite
```
@misc{thiry2022modified,
  doi = {10.48550/ARXIV.2204.13914},
  url = {https://arxiv.org/abs/2204.13914},
  author = {Thiry, Louis and Li, Long and Mémin, Etienne},
  keywords = {Fluid Dynamics (physics.flu-dyn), Geophysics (physics.geo-ph), FOS: Physical sciences, FOS: Physical sciences},
  title = {Modified (hyper-)viscosity for coarse-resolution ocean models},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
