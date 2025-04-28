==========
SpectroPrep
==========

.. image:: https://img.shields.io/pypi/v/spectoprep.svg
        :target: https://pypi.python.org/pypi/spectoprep

.. image:: https://img.shields.io/travis/habeeb3579/spectoprep.svg
        :target: https://travis-ci.com/habeeb3579/spectoprep

.. image:: https://readthedocs.org/projects/spectoprep/badge/?version=latest
        :target: https://spectoprep.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

Spectroscopy preprocessing using Bayesian Optimization

Overview
--------

SpectroPrep provides a toolkit for optimizing spectroscopic data preprocessing pipelines using Bayesian optimization. It automatically discovers the optimal combination of preprocessing techniques and their parameters to improve model performance for spectroscopic data analysis.

Features
--------

- **Pipeline Optimization**: Automate the discovery of optimal preprocessing pipelines using Bayesian optimization
- **Flexible Preprocessing**: Choose from multiple preprocessing techniques (MSC, SNV, Savitzky-Golay, etc.)
- **Cross-Validation Support**: Group-based cross-validation methods for robust evaluation
- **Configurable Pipeline Length**: Control maximum preprocessing steps and allowed combinations

Installation
-----------

.. code-block:: bash

    pip install spectroprep

Quick Start
----------

.. code-block:: python

    from spectroprep.pipeline.optimizer import PipelineOptimizer
    import numpy as np
    
    # Prepare your data
    X_train = np.array(...)  # Your spectral data matrix
    y_train = np.array(...)  # Your target values
    groups = np.array(...)   # Optional group labels for cross-validation
    
    # Initialize the optimizer
    optimizer = PipelineOptimizer(
        X_train=X_train,
        y_train=y_train,
        X_test=None,
        y_test=None,
        preprocessing_steps=['msc', 'savgol', 'detrend', 'scaler', 'snv', 
                            'robust_scaler', 'emsc', 'meancn'],
        cv_method="group_shuffle_split",
        n_splits=3,
        random_state=21,
        groups=groups,
        max_pipeline_length=2,
        allowed_preprocess_combinations=[1, 2]
    )
    
    # Run Bayesian optimization to find the best pipeline
    best_params, best_pipeline = optimizer.bayesian_optimize(
        init_points=50,
        n_iter=1000
    )
    
    # Extract preprocessing steps without the final model
    custom_preprocessing = []
    for name, step in best_pipeline.steps[:-1]:
        custom_preprocessing.append((name, step))
    
    # Print optimization summary
    summary = optimizer.summarize_optimization()
    print(f"Best pipeline configuration: {summary['best_pipeline']}")
    print(f"Best RMSE: {summary['best_rmse']:.4f}")
    
    # Make predictions with the optimized pipeline
    predictions, rmse, r2 = optimizer.get_best_pipeline_predictions(best_pipeline)

Available Preprocessing Methods
------------------------------

- **msc**: Multiplicative Scatter Correction
- **savgol**: Savitzky-Golay filtering
- **detrend**: Linear detrending
- **scaler**: Standard scaling
- **snv**: Standard Normal Variate
- **robust_scaler**: Robust scaling
- **emsc**: Extended Multiplicative Signal Correction
- **meancn**: Mean centering
- **pca**: Principal Component Analysis
- **select_k_best**: Feature selection

Documentation
------------

For detailed documentation, visit `spectroprep.readthedocs.io <https://spectroprep.readthedocs.io>`_.

Contributing
-----------

We welcome contributions! Please feel free to submit a Pull Request.

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage