=====
Usage
=====

This page summarises the public Python API and the command-line interface.
For a full walk-through, see :doc:`getting_started`.

Python API
----------

Core imports::

   from spectoprep import (
       PipelineOptimizer,
       OptimizedRidgeCV,
       SpectroPrepPlotter,
       configure_logging,
       get_logger,
   )

PipelineOptimizer
~~~~~~~~~~~~~~~~~

:class:`~spectoprep.PipelineOptimizer` is the main entry point.

1. Supply training spectra ``X_train`` (``n_samples × n_wavelengths``) and
   targets ``y_train``.
2. Optionally pass ``groups`` so replicates from the same sample stay in the
   same CV fold.
3. Choose preprocessing candidates and a CV strategy.
4. Call :meth:`~spectoprep.PipelineOptimizer.bayesian_optimize`.
5. Inspect results with :meth:`~spectoprep.PipelineOptimizer.summarize_optimization`
   or :meth:`~spectoprep.PipelineOptimizer.get_best_pipeline_predictions`.

.. code-block:: python

   optimizer = PipelineOptimizer(
       X_train=X_train,
       y_train=y_train,
       groups=groups,
       preprocessing_steps=["msc", "savgol", "detrend", "snv", "scaler"],
       cv_method="group_kfold",   # or group_shuffle_split / leave_p_group_out
       n_splits=5,
       max_pipeline_length=2,
       allowed_preprocess_combinations=[1, 2],
       log_level="INFO",
   )

   best_params, best_pipeline = optimizer.bayesian_optimize(
       init_points=25,
       n_iter=200,
       acquisition_function="ei",
   )

   summary = optimizer.summarize_optimization()
   preds, rmse, r2 = optimizer.get_best_pipeline_predictions(best_pipeline)

Important design notes
~~~~~~~~~~~~~~~~~~~~~~

* The final estimator is always :class:`~spectoprep.OptimizedRidgeCV`. The
  ridge penalty is selected **inside** each objective evaluation; there is no
  ``ridge_alpha`` hyperparameter in the Bayesian search space.
* The Bayesian objective scores **training-fold CV only**. Held-out
  ``X_test`` / ``y_test`` are used for reporting after optimization, not for
  selecting the pipeline (avoids test leakage).
* Incompatible transforms (for example multiple scatter corrections) are
  filtered via configuration in :mod:`spectoprep.pipeline.config`.

Cross-validation strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

======================= =======================================================
``cv_method``           Behaviour
======================= =======================================================
``group_kfold``         Deterministic :class:`~sklearn.model_selection.GroupKFold`
``group_shuffle_split`` Random :class:`~sklearn.model_selection.GroupShuffleSplit`
``leave_p_group_out``   :class:`~sklearn.model_selection.LeavePGroupsOut`
======================= =======================================================

If ``groups`` is omitted, each sample becomes its own group.

Preprocessing catalogue
~~~~~~~~~~~~~~~~~~~~~~~

Call the CLI or inspect :data:`spectoprep.pipeline.config.AVAILABLE_STEPS`.
Common keys include ``msc``, ``emsc``, ``snv``, ``savgol``, ``detrend``,
``als``, ``scaler``, ``robust_scaler``, ``meancn``, ``pca``, and
``select_k_best``.

Logging
~~~~~~~

SpectoPrep uses :mod:`structlog`. Library imports do not configure logging;
the optimizer and CLI call :func:`~spectoprep.configure_logging` at entry::

   from spectoprep import configure_logging, get_logger

   configure_logging(level="DEBUG", json_logs=False)
   log = get_logger("my_app")
   log.info("run_started", n_samples=X_train.shape[0])

Command-line interface
----------------------

After installation the ``spectoprep`` console script is available:

.. code-block:: console

   $ spectoprep version
   $ spectoprep info
   $ spectoprep --log-level DEBUG --json-logs info

=========== ===========================================================
Command     Description
=========== ===========================================================
``version`` Print the installed package version
``info``    List registered preprocessing transforms
=========== ===========================================================

Visualization
-------------

:class:`~spectoprep.SpectroPrepPlotter` is the supported way to inspect runs.
Typical post-optimisation workflow::

   SpectroPrepPlotter.set_style()
   SpectroPrepPlotter.plot_spectra(wavelengths, X_train[:20])
   SpectroPrepPlotter.plot_preprocessing_comparison(
       wavelengths, X_train, {"prep": X_prep}, sample_indices=[0, 1]
   )
   SpectroPrepPlotter.plot_prediction_scatter(y_true, y_pred)
   SpectroPrepPlotter.plot_optimization_progress(optimizer)
   SpectroPrepPlotter.plot_optimization_results(optimizer, top_n=5)

Every method accepts an optional ``save_path`` to write a PNG. Full signatures
are in :doc:`api/visualization`. A self-contained demo is
``examples/plotting_demo.py``.
