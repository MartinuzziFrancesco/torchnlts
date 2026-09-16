# Tutorials

Hand-holding walkthroughs of library functionality, each reproducing a
specific published result as a runnable check that the implementation
matches the literature (not just the unit tests). One script per
result; add one whenever a new module lands a result worth reproducing.

For harder technical showcases (advanced, composed usage rather than
guided walkthroughs), see `../examples/`.

- `delay_embedding_lorenz.py`: Takens (1981) -- a scalar observable's
  delay embedding is diffeomorphic to the original attractor.
- `numpy_pandas_interop.py`: using torchnlts functions directly on
  numpy/pandas data, and converting results back with
  `torchnlts.convert`.
