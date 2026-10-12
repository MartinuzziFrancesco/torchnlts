torchnlts
=========

Welcome to the **torchnlts** documentation!

**torchnlts** is a differentiable nonlinear time series analysis library
built on PyTorch: classical dynamical-systems measures (recurrence,
correlation dimension, Lyapunov exponents, entropy, causality) exposed
as ordinary, autodiff-compatible tensor operations.

Highlights
----------

- **Core shared primitives** (pairwise distance, Theiler-window masking,
  neighborhood search, delay embedding, trajectory kernels) validated
  against both the defining paper and DynamicalSystems.jl
- **Explicit gradient-class taxonomy**: every function states whether
  it's exact, a smooth relaxation, piecewise, or analysis-only
- **PyTorch-native interface**: ``[T]``, ``[T, D]``, or ``[B, T, D]``
  tensors in, CPU/CUDA/float32/float64 throughout
- **numpy/pandas interop**: every function also accepts numpy arrays and
  pandas Series/DataFrame; see :doc:`api/convert` for converting
  results back out

Installation
------------

.. code-block:: bash

   pip install torchnlts

Quick Example
-------------

.. code-block:: python

   import torch
   from torchnlts.neighbors import knn_search
   from torchnlts.embedding import delay_embedding

   trajectory = torch.randn(500)
   embedded = delay_embedding(trajectory, embedding_dim=3, delay=5)
   distances, indices = knn_search(embedded, num_neighbors=4, theiler_window=5)

.. toctree::
   :titlesonly:
   :hidden:
   :maxdepth: 2

   primitives
   api/index
