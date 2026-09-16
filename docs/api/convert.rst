Convert Reference
==================

numpy/pandas interoperability -- ``torchnlts.convert``. ``as_tensor`` is
wired into every trajectory-consuming function automatically;
``to_numpy``/``to_pandas`` are explicit, opt-in output conversions (never
automatic, so autodiff is never silently broken).

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   torchnlts.convert.as_tensor
   torchnlts.convert.to_numpy
   torchnlts.convert.to_pandas
