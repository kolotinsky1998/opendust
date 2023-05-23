# OpenDust
OpenDust: A fast GPU-accelerated code for calculation forces, acting on microparticles in a plasma flow
<img src="https://github.com/kolotinsky1998/opendust/blob/main/animation/animation.gif" width="600" height="450" />

Opendust is GPU-based, Python library for dusty plasmas. 

OpenDust aims to provide researchers both experimenters and theorists user-friendly and high-performance tool for calculation forces, acting on microparticles, and microparticles’ charges in a plasma flow. 

OpenDust performance originates from highly-optimized Cuda back-end and allows to perform self-consistent calculation of plasma flow around microparticles in seconds.

## Installation
OpenDust is installed using [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

1. Begin by installing the most recent 64 bit, Python 3.x version of Miniconda
2. For using OpenDust you need Nvidia GPU. Make sure you have installed modern GPU [drivers](https://www.nvidia.com/Download/index.aspx). CUDA itself will be provided by the `cudatoolkit` package when you install `opendust` in the next steps.
3. Open a command line terminal and type the following commands
```
conda create -n opendust
```
```
conda activate opendust
```
```
conda install -c kolotinsky -c conda-forge opendust
```
With recent conda versions (v4.8.4+), this will install a version of OpenDust compiled with the latest version of CUDA supported by your drivers. Alternatively you can request a version that is compiled for a specific CUDA version with the command
```
conda install -c kolotinsky -c conda-forge opendust cudatoolkit=10.0
```
where `10.0` should be replaced with the particular CUDA version you want to target.
##  Documentation
https://github.com/kolotinsky1998/opendust/tree/main/tutorial

##  How to cite
In case of usage OpenDust please cite the article:

Kolotinskii D., Timofeev A. OpenDust: A fast GPU-accelerated code for the calculation of forces acting on microparticles in a plasma flow //Computer Physics Communications. – 2023. – Т. 288. – С. 108746.

##  Author's contacts

If you have any questions, comments or offers related to the OpenDust package or the base article do not to hesitate to contact me!

Email: kolotinskiy.da@phystech.edu

Research Gate: https://www.researchgate.net/profile/Daniil_Kolotinskii

Telegram: @daniil_kolotinskii
