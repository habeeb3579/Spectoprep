SpectoPrep
==========

.. image:: https://img.shields.io/pypi/v/spectoprep.svg
    :target: https://pypi.python.org/pypi/spectoprep
    :alt: PyPI version

.. image:: https://github.com/habeeb3579/Spectoprep/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/habeeb3579/Spectoprep/actions/workflows/ci.yml
    :alt: CI status

.. image:: https://codecov.io/gh/habeeb3579/Spectoprep/graph/badge.svg?token=5EPSYE77K7
    :target: https://codecov.io/gh/habeeb3579/Spectoprep
    :alt: Coverage

.. image:: https://anaconda.org/habeebest/spectoprep/badges/version.svg
    :target: https://anaconda.org/habeebest/spectoprep
    :alt: Conda version

.. image:: https://readthedocs.org/projects/spectoprep/badge/?version=latest
    :target: https://spectoprep.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Bayesian optimization of spectroscopic preprocessing pipelines for chemometric
**regression** (NIR, MIR, Raman, and related modalities).

Overview
--------

SpectoPrep searches over combinations of scatter correction, smoothing,
baseline correction, scaling and related transforms. Each candidate pipeline is
scored with group-aware cross-validation and fitted with
``OptimizedRidgeCV``, which selects the ridge penalty by CV. The goal is a
reproducible, leakage-aware alternative to hand-tuned preprocessing recipes.

Features
--------

- **Bayesian pipeline search** over structure and transform hyperparameters
- **Group-aware CV**: ``group_kfold``, ``group_shuffle_split``, ``leave_p_group_out``
- **RidgeCV downstream model** (no manual ``ridge_alpha`` in the search space)
- **Broad preprocessing catalogue** (MSC, EMSC, SNV, Savitzky–Golay, ALS, scalers, PCA, …)
- **Structured logging** via ``structlog`` and a Typer CLI (``spectoprep info``)
- **Visualization helpers** for spectra and optimization summaries

Installation
------------

.. code-block:: bash

    pip install spectoprep

Or::

    conda install -c habeebest spectoprep

PyPI and the ``habeebest`` conda channel are both published automatically from
tagged releases (see ``updated_package_deployment.md``).

Requires Python 3.10+.

Quick start
-----------

.. code-block:: python

    import numpy as np
    from spectoprep import PipelineOptimizer

    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(80, 200))
    y_train = rng.normal(size=80)
    groups = np.arange(80)

    optimizer = PipelineOptimizer(
        X_train=X_train,
        y_train=y_train,
        groups=groups,
        preprocessing_steps=["msc", "savgol", "detrend", "scaler", "snv"],
        cv_method="group_kfold",
        n_splits=5,
        max_pipeline_length=2,
        allowed_preprocess_combinations=[1, 2],
    )

    best_params, best_pipeline = optimizer.bayesian_optimize(
        init_points=25,
        n_iter=200,
    )

    summary = optimizer.summarize_optimization()
    predictions, rmse, r2 = optimizer.get_best_pipeline_predictions(best_pipeline)

CLI
---

.. code-block:: bash

    spectoprep version
    spectoprep info

Selected preprocessing methods
------------------------------

- **msc** / **emsc**: multiplicative (extended) scatter correction
- **snv** / **lsnv** / **rnv**: (localized / robust) standard normal variate
- **savgol**: Savitzky–Golay filtering / derivatives
- **detrend** / **als**: linear detrend / asymmetric least squares baseline
- **scaler** / **robust_scaler** / **meancn**: column scaling and mean centering
- **pca** / **select_k_best**: dimensionality reduction and feature selection

Run ``spectoprep info`` for the full catalogue.

Documentation
-------------

- Read the Docs: https://spectoprep.readthedocs.io
- GitHub Pages: https://habeeb3579.github.io/Spectoprep/

Limitations
-----------

SpectoPrep currently supports **regression** only. Classification pipelines
are out of scope.

Contributing
------------

Contributions are welcome. See ``CONTRIBUTING.rst`` and open a pull request.

License
-------

MIT — see ``LICENSE``.

Citation
--------

If you use SpectoPrep in research, please cite:

.. code-block:: bibtex

   @article{babatunde2025automated,
     title={Automated Spectral Preprocessing via Bayesian Optimization for Chemometric Analysis of Milk Constituents},
     author={Babatunde, Habeeb Abolaji and McDougal, Owen M and Andersen, Timothy},
     journal={Foods},
     volume={14},
     number={17},
     pages={2996},
     year={2025},
     publisher={MDPI}
   }
