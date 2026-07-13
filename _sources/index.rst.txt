SpectoPrep documentation
========================

**SpectoPrep** finds effective spectroscopic preprocessing pipelines with
Bayesian optimization. It searches over scatter correction, smoothing,
baseline, scaling and related transforms, then fits a ridge model whose
penalty is selected by cross-validation.

.. code-block:: python

   from spectoprep import PipelineOptimizer
   import numpy as np

   rng = np.random.default_rng(0)
   X = rng.normal(size=(80, 200))
   y = rng.normal(size=80)
   groups = np.arange(80)

   optimizer = PipelineOptimizer(
       X_train=X,
       y_train=y,
       groups=groups,
       preprocessing_steps=["msc", "savgol", "snv", "scaler"],
       cv_method="group_kfold",
       n_splits=5,
       max_pipeline_length=2,
       allowed_preprocess_combinations=[1, 2],
   )
   best_params, best_pipeline = optimizer.bayesian_optimize(init_points=5, n_iter=20)

Why SpectoPrep?
---------------

* **Joint search** — pipeline structure and transform hyperparameters in one
  Bayesian loop.
* **Group-aware CV** — ``group_kfold``, ``group_shuffle_split``, or
  ``leave_p_group_out`` so replicate spectra stay together.
* **Modern ridge backend** — downstream
  :class:`~spectoprep.OptimizedRidgeCV` selects ``alpha`` by CV (no manual
  ``ridge_alpha`` search).
* **Production tooling** — structured logging (``structlog``), a Typer CLI,
  and a full test suite.

Scope
-----

SpectoPrep targets **regression** chemometric workflows (NIR, MIR, Raman and
similar). Classification is not supported.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   usage
   getting_started
   notebooks/corn_benchmark
   notebooks/tutorial

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
   modules

.. toctree::
   :maxdepth: 1
   :caption: Project

   readme
   contributing
   authors
   history

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
