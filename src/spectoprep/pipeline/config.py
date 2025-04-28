"""
Configuration constants for the ML Pipeline Optimizer.
"""
#autoscaling is same as standardscaler or column-wise standardization
#SNV is same as row standardization

# Dictionary of available preprocessing steps
AVAILABLE_STEPS = {
    # Original steps
    'als': 'Asymmetric Least Squares',
    'savgol': 'Savitzky-Golay Filter',
    'snv': 'Standard Normal Variate',
    'normalization': 'Normalization',
    'scaler': 'Standard Scaler',
    'pca': 'Principal Component Analysis',
    'detrend': 'Detrend',
    'msc': 'Multiplicative Scatter Correction',
    'emsc': 'Extended Multiplicative Scatter Correction',
    'autoscale': 'Autoscaling',
    'globalscale': 'Global Scaling',
    'lsnv': 'Localized SNV',
    'rnv': 'Robust Normal Variate',
    'meancn': 'MeanCentering',
    
    # Additional sklearn steps
    'robust_scaler': 'Robust Scaler',
    'minmax_scaler': 'Min-Max Scaler',
    'power_transformer': 'Power Transformer',
    'quantile_transformer': 'Quantile Transformer',
    'row_standardizer': 'Row Standardizer',
    'fast_ica': 'Fast ICA',
    'kernel_pca': 'Kernel PCA',
    'lle': 'Locally Linear Embedding',
    'select_k_best': 'Select K Best Features',
    'feature_agglomeration': 'Feature Agglomeration',
    'rbf_sampler': 'RBF Kernel Approximation'
}

# Define groups of steps that should not be used together
INCOMPATIBLE_SETS = [
    # Scatter correction methods
    ['snv', 'msc', 'emsc', 'lsnv', 'rnv'],
    
    # Scaling methods (expanded)
    ['scaler', 'autoscale', 'globalscale', 'normalization', 'robust_scaler', 'minmax_scaler', 
     'power_transformer', 'quantile_transformer', 'row_standardizer'],

    ['snv', 'row_standardizer'],

    ['autoscale', 'scaler'],

    ['robust_scaler', 'rnv'],
    
    # Dimensionality reduction methods
    ['pca', 'fast_ica', 'kernel_pca', 'lle'],
    
    # Feature selection/reduction methods
    ['select_k_best', 'feature_agglomeration']
]

# Default parameter bounds for Bayesian optimization
DEFAULT_PARAM_BOUNDS = {
    # Original parameters
    "als_lam": (1e2, 1e5),
    "als_p": (0.01, 0.3),
    "als_niter": (10, 50),
    "savgol_filter_win": (7, 13),
    "savgol_poly_order": (1, 5),
    "savgol_deriv_order": (0, 3),
    "detrend_method": (0, 1),
    "detrend_order": (1, 5),
    "emsc_order": (1, 5),
    "global_factor": (1, 10),
    "lsnv_win": (7, 13),
    "rnv_lp": (5, 20),
    "rnv_up": (70, 95),
    "pca_n_components": (2, 100),  # Will be capped by feature count
    
    # New parameters for additional preprocessors
    "rs_quantile_low": (5, 20),
    "rs_quantile_high": (70, 95),
    "rs_with_centering": (0, 1),
    "rs_with_scaling": (0, 1),
    "minmax_min": (0, 0.4),
    "minmax_max": (0.6, 1),
    "power_method": (0, 1),
    "power_standardize": (0, 1),
    "quantile_n": (10, 200),
    "quantile_dist": (0, 1),
    "ica_n_components": (2, 50),  # Will be capped by feature count
    "kpca_n_components": (2, 50),  # Will be capped by feature count
    "kpca_kernel": (0, 3.99),  # Will be rounded to 0, 1, 2, or 3
    "kpca_gamma": (0.001, 10),
    "lle_n_components": (2, 30),  # Will be capped by feature count
    "lle_n_neighbors": (5, 30),  # Will be capped by sample count - 1
    "skb_k": (2, 100),  # Will be capped by feature count
    "fa_n_clusters": (2, 100),  # Will be capped by feature count - 1
    "fa_linkage": (0, 2.99),  # Will be rounded to 0, 1, or 2
    "rbf_n_components": (10, 200),
    "rbf_gamma": (0.001, 1.0),
    
    # Ridge regression parameter
    "ridge_alpha": (0.001, 100.0)
}