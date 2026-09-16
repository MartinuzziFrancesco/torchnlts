"""Using torchnlts with numpy/pandas data, not just tensors.

Every torchnlts function accepts a numpy array or pandas Series/DataFrame
directly -- no manual `torch.as_tensor` needed. Results come back as
tensors (so autodiff keeps working); convert them back explicitly with
`torchnlts.convert.to_numpy` / `to_pandas` when you're done computing
and just want plain data again.
"""

import pandas as pd

from torchnlts.convert import to_pandas
from torchnlts.embedding.delay import delay_embedding
from torchnlts.neighbors.knn import knn_search

# A pandas Series, as you'd get from reading a CSV column.
series = pd.Series([0.0, 1.0, 2.0, 3.5, 10.0, 10.2, 20.0], name="sensor_reading")

embedded = delay_embedding(series, embedding_dim=2, delay=1)
distances, indices = knn_search(embedded, num_neighbors=2, theiler_window=1)

# Back to pandas for downstream analysis, plotting, or a CSV write.
embedded_df = to_pandas(embedded, columns=["x(t)", "x(t+1)"])
print(embedded_df)
print()
print("nearest-neighbor distances:")
print(to_pandas(distances))
