===============
Getting started
===============

This guide runs a small, fully reproducible example with synthetic spectra.
It mirrors the public API used in the SoftwareX manuscript (nested CV on real
NIR data is described there; here we keep runtime short for documentation).

1. Imports and synthetic data
-----------------------------

.. code-block:: python

   import numpy as np
   from spectoprep import PipelineOptimizer, SpectroPrepPlotter

   rng = np.random.default_rng(21)
   n_samples, n_wavelengths = 60, 100
   wavelengths = np.linspace(1100, 2500, n_wavelengths)

   # Mild Beer–Lambert-like peaks + scatter + noise
   baseline = 0.2 + 0.001 * (wavelengths - wavelengths.mean())
   peaks = (
       0.8 * np.exp(-0.5 * ((wavelengths - 1450) / 40) ** 2)
       + 0.5 * np.exp(-0.5 * ((wavelengths - 1900) / 55) ** 2)
   )
   concentrations = rng.uniform(0.5, 1.5, size=n_samples)
   scatter = rng.normal(1.0, 0.05, size=(n_samples, 1))
   X = scatter * (baseline + np.outer(concentrations, peaks))
   X += rng.normal(0.0, 0.01, size=X.shape)
   y = concentrations + rng.normal(0.0, 0.02, size=n_samples)
   groups = np.arange(n_samples)  # one spectrum per sample

2. Configure and run the optimizer
----------------------------------

Keep the search space small for a quick demo. Prefer ``group_kfold`` when
each physical sample has a group label.

.. code-block:: python

   optimizer = PipelineOptimizer(
       X_train=X,
       y_train=y,
       groups=groups,
       preprocessing_steps=["msc", "savgol", "snv", "scaler", "detrend"],
       cv_method="group_kfold",
       n_splits=5,
       max_pipeline_length=2,
       allowed_preprocess_combinations=[1, 2],
       random_state=21,
       log_level="WARNING",  # quieter for notebooks
   )

   best_params, best_pipeline = optimizer.bayesian_optimize(
       init_points=8,
       n_iter=25,
   )

3. Inspect results
------------------

.. code-block:: python

   summary = optimizer.summarize_optimization()
   print(summary["best_pipeline"])
   print(f"Best CV score (neg-RMSE): {summary['best_rmse']:.4f}")

   preds, rmse, r2 = optimizer.get_best_pipeline_predictions(best_pipeline)
   print(f"Train RMSE={rmse:.4f}, R²={r2:.4f}")
   print(best_pipeline)

The returned ``best_pipeline`` is a scikit-learn
:class:`~sklearn.pipeline.Pipeline` ending in
:class:`~spectoprep.OptimizedRidgeCV`. You can ``joblib.dump`` it, or strip
the final estimator and reuse only the preprocessing steps.

4. Visualise with SpectroPrepPlotter
------------------------------------

Use the package plotters rather than ad-hoc Matplotlib calls:

.. code-block:: python

   from sklearn.pipeline import Pipeline

   SpectroPrepPlotter.set_style(context="paper", font_scale=1.1)
   SpectroPrepPlotter.plot_spectra(
       wavelengths,
       X[:12],
       title="Synthetic NIR-like spectra",
       xlabel="Wavelength (nm)",
       ylabel="Absorbance (a.u.)",
   )

   prep = Pipeline(best_pipeline.steps[:-1])
   X_prep = prep.transform(X)
   SpectroPrepPlotter.plot_preprocessing_comparison(
       wavelengths,
       X,
       {"Best pipeline": X_prep},
       sample_indices=[0, 1, 2],
       title="Raw vs. optimised preprocessing",
   )

   SpectroPrepPlotter.plot_prediction_scatter(
       y, preds, title="Predicted vs. reference",
       xlabel="Reference", ylabel="Predicted",
   )
   SpectroPrepPlotter.plot_optimization_progress(optimizer)
   SpectroPrepPlotter.plot_optimization_results(optimizer, top_n=5)

A runnable script that writes these figures to disk ships with the repository:

.. code-block:: console

   $ python examples/plotting_demo.py

Next steps
----------

* Browse the full transform list with ``spectoprep info``.
* For a public NIR end-to-end example (spectra, consensus preprocessing,
  predicted-versus-reference), see the :doc:`notebooks/corn_benchmark` notebook
  and ``examples/run_corn_benchmark.py``.
* The legacy milk FTIR notebook remains at :doc:`notebooks/tutorial`.
* For production calibrations, increase ``init_points`` / ``n_iter`` and use
  nested cross-validation so every sample is predicted while held out.
* See :doc:`api/visualization` for every plotter method and :doc:`usage` for CV
  and logging details.
