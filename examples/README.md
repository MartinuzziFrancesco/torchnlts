# Examples

Each script reproduces a specific published result using torchnlts, as
a runnable check that the implementation matches the literature (not
just the unit tests). One script per result; add one whenever a new
module lands a result worth reproducing.

- `delay_embedding_lorenz.py`: Takens (1981) -- a scalar observable's
  delay embedding is diffeomorphic to the original attractor.
