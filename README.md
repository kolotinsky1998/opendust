# OpenDust
OpenDust: A fast GPU-accelerated code for calculation forces, acting on micro-particles in a plasma flow
<img src="https://github.com/kolotinsky1998/opendust/blob/main/animation/animation.gif" width="600" height="450" />
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
## Installation
https://github.com/kolotinsky1998/opendust/tree/main/tutorial
