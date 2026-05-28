# jaxFMM

jaxFMM is an open source implementation of the Fast Multipole Method in JAX. The goal is to offer an easily readable/maintainable FMM implementation with good performance that runs on CPU/GPU and supports autodiff. This is enabled through JAX's just-in-time compiler.

## Installation and Usage

jaxFMM depends only on JAX and can be installed from pypi or by downloading the source as follows:

    pip install jaxfmm

If you want to run jaxFMM on GPUs, the easiest way is to use NVIDIA CUDA and cuDNN from pip wheels by instead typing:

    pip install jaxfmm[cuda]

Using a custom, self-installed CUDA with jax is [described in the JAX documentation](https://docs.jax.dev/en/latest/installation.html).

The [unitcube demo](/demos/unitcube.py) is a short and simple example demonstrating how to use jaxFMM.

## Features

There are many flavors of FMM implementations. In short, jaxFMM currently:

- only supports point charges and the Laplacian kernel.
- uses real basis functions computed via recurrence relations.
- uses rotation-based O(p^3) M2M/M2L/L2L transformations by computing rotation matrices on-the-fly with fast recursions.
- uses a non-uniform 2^N-ary tree hierarchy (directly inspired by [a work of A. Goude and S. Engblom](https://link.springer.com/article/10.1007/s11227-012-0836-0)), allowing arbitrary shape of the boxes in the hierarchy and guaranteeing balanced trees but requiring storage of interaction lists.
- has jit-compiled functions and autodiff for every substep of the algorithm except for the generation of interaction lists.

In summary, jaxFMM in its current state can do adaptive point charge FMM for Laplace kernels with good performance for constant particle positions and reasonably homogenous distributions. For more details, consider reading the [preprint on arXiv](https://arxiv.org/abs/2511.15269).

## Citing

A preprint describing jaxFMM is [available on arXiv](https://arxiv.org/abs/2511.15269), which can be cited with the following BibTeX entry:

    @misc{kraft2025jaxfmmadaptivegpuparallelimplementation,
        title={jaxFMM: An Adaptive, GPU-Parallel Implementation of the Fast Multipole Method in JAX}, 
        author={Robert Kraft and Florian Bruckner and Dieter Suess and Claas Abert},
        year={2025},
        eprint={2511.15269},
        archivePrefix={arXiv},
        primaryClass={physics.comp-ph},
        url={https://arxiv.org/abs/2511.15269}, 
    }

## TODOs

jaxFMM is primarily developed for my PhD project, where I am working on a GPU FMM stray field evaluation routine for finite-element micromagnetics. This explains the feature set and design decisions mentioned above.

Contributions are always welcome however and improvements to jaxFMM are in development, such as:

- distributed parallelism via jax.sharding.
- varying parameters locally for better performance (particularly the number of splits per axis and level).
- investigating kernel-independent/volume FMM.

## Stray Field Evaluation

As mentioned above, jaxFMM is developed for rapid stray field evaluation in finite-element micromagnetics. The [strayfield branch](https://gitlab.com/jaxfmm/jaxfmm/-/tree/strayfield) features stray field evaluation functions for P1-FEM meshes and the corresponding [strayfield unitcube demo](https://gitlab.com/jaxfmm/jaxfmm/-/blob/strayfield/demos/strayfield_unitcube.py) shows how they are used. Note that this is still work in progress and currently only gives accurate results for meshes of good quality (i.e. all tetrahedra have low aspect ratio).