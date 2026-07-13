Examples
========

``run_corn_benchmark.py``
  Paper-grade nested five-fold CV on the Cargill corn NIR ``m5`` dataset for
  **Moisture, Oil, Protein, Starch**. Writes:

  * ``examples/corn_results/optimal_pipelines.json`` — per-fold and consensus
    preprocessing chains + metrics
  * ``examples/corn_results/figures/prediction_*.png`` — predicted vs reference
  * ``examples/corn_results/figures/spectra_*.png`` — full raw / preprocessed
    spectra with **inverted** wavelength axis

.. code-block:: console

   python examples/run_corn_benchmark.py

``plotting_demo.py``
  Short synthetic SpectoPrepPlotter demo.

``make_paper_prediction_figure.py``
  Protein-only nested-CV scatter (subset of the full corn benchmark).

Notebooks
---------

* ``docs/notebooks/corn_benchmark.ipynb`` — corn walk-through (preferred)
* ``docs/notebooks/tutorial.ipynb`` — legacy milk FTIR example
