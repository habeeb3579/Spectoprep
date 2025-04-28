"""
Functions for building preprocessing steps from parameters.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.cluster import FeatureAgglomeration
from sklearn.kernel_approximation import RBFSampler
from sklearn.manifold import LocallyLinearEmbedding

# Import custom preprocessing components
try:
    from spectoprep.preprocessing import *
    from .utils import *
except ImportError:
    # Fallback to direct imports if package structure is not available
    from ..preprocessing.baseline import ALSBaselineCorrection, DetrendTransformer
    from ..preprocessing.scatter import SNV, MSC, EMSC, LocalizedSNV, RobustNormalVariate
    from ..preprocessing.norml import Normalization, Autoscaling, GlobalScaling, RowStandardizer
    from ..preprocessing.smoothing import SavgolFilter


def build_preprocessor_from_bayes(name: str, params: Dict, X_train_shape: tuple, random_state: int, n_jobs: int) -> object:
    """
    Build a transformer from bayesian optimization parameters.
    
    Args:
        name: Name of the transformer
        params: Dictionary of parameters for the transformer
        X_train_shape: Shape of the training data
        random_state: Random state for reproducibility
        n_jobs: Number of CPU cores to use
        
    Returns:
        The constructed transformer object
    """
    if name == "als":
        return ALSBaselineCorrection(
            lam=params["als_lam"],
            p=params["als_p"],
            niter=int(round(params["als_niter"]))
        )
    elif name == "savgol":
        filter_win = choose_nearest(params["savgol_filter_win"], [7, 9, 11, 13])
        return SavitzkyGolay(
            filter_win=filter_win,
            poly_order=int(round(params["savgol_poly_order"])),
            deriv_order=int(round(params["savgol_deriv_order"]))
        )
    elif name == "snv":
        return StandardNormalVariate()
    elif name == "lsnv":
        lsnv_win = choose_nearest(params["lsnv_win"], [7, 9, 11, 13])
        return LocalizedSNV(window_size=lsnv_win)
    elif name == "rnv":
        rnv_lp = choose_nearest(params["rnv_lp"], [5, 10, 15, 20])
        rnv_up = choose_nearest(params["rnv_up"], [80, 85, 90, 95])
        return RobustNormalVariate(lower_percentile=rnv_lp, upper_percentile=rnv_up)
    elif name == "normalization":
        # For simplicity, fix normalization parameters
        method = "minmax"
        return Normalization(method=method, feature_range=(0, 1))
    elif name == "detrend":
        method = "simple" if int(round(params["detrend_method"])) == 0 else "polynomial"
        return DetrendTransformer(method=method, order=int(round(params["detrend_order"])))
    elif name == "msc":
        return MultiplicativeScatterCorrection()
    elif name == "emsc":
        return ExtendedMultiplicativeScatterCorrection(order=int(round(params["emsc_order"])))
    elif name == "autoscale":
        return Autoscaling()
    elif name == "globalscale":
        return GlobalScaler(factor=int(round(params["global_factor"])))
    elif name == "meancn":
        return MeanCentering()
    elif name == "scaler":
        return StandardScaler()
    elif name == "pca":
        return PCA(
            n_components=min(int(round(params["pca_n_components"])), X_train_shape[1]), 
            random_state=random_state
        )
    
    # New preprocessors
    elif name == "robust_scaler":
        quantile_range = (
            choose_nearest(params["rs_quantile_low"], [5, 10, 15, 20, 25]),
            choose_nearest(params["rs_quantile_high"], [75, 80, 85, 90, 95])
        )
        return RobustScaler(
            with_centering=params["rs_with_centering"] > 0.5,
            with_scaling=params["rs_with_scaling"] > 0.5,
            quantile_range=quantile_range
        )
    elif name == "minmax_scaler":
        return MinMaxScaler(
            feature_range=(params["minmax_min"], params["minmax_max"])
        )
    elif name == "power_transformer":
        method = "yeo-johnson" if params["power_method"] > 0.5 else "box-cox"
        return PowerTransformer(
            method=method,
            standardize=params["power_standardize"] > 0.5
        )
    elif name == "quantile_transformer":
            n_quantiles = min(int(round(params["quantile_n"])), X_train_shape[0])
            output_dist = "normal" if params["quantile_dist"] > 0.5 else "uniform"
            return QuantileTransformer(
                n_quantiles=max(10, n_quantiles),
                output_distribution=output_dist,
                random_state=random_state
            )
    elif name == "row_standardizer":
        return RowStandardizer()
    elif name == "fast_ica":
        n_components = int(round(params["ica_n_components"]))
        n_components = min(n_components, X_train_shape[1])
        return FastICA(
            n_components=n_components,
            random_state=random_state,
            max_iter=1000
        )
    elif name == "kernel_pca":
        n_components = int(round(params["kpca_n_components"]))
        n_components = min(n_components, X_train_shape[1])
        kernel = ["linear", "poly", "rbf", "sigmoid"][int(round(params["kpca_kernel"]))]
        return KernelPCA(
            n_components=n_components,
            kernel=kernel,
            gamma=params["kpca_gamma"],
            random_state=random_state,
            n_jobs=n_jobs
            )
    elif name == "lle":
        n_components = int(round(params["lle_n_components"]))
        n_components = min(n_components, X_train_shape[1])
        n_neighbors = int(round(params["lle_n_neighbors"]))
        n_neighbors = min(n_neighbors, X_train_shape[0] - 1)
        return LocallyLinearEmbedding(
            n_components=n_components,
            n_neighbors=max(5, n_neighbors),
            random_state=random_state,
            n_jobs=n_jobs
        )
    elif name == "select_k_best":
        k = int(round(params["skb_k"]))
        k = min(k, X_train_shape[1])
        return SelectKBest(
            f_regression,
            k=max(1, k)
        )
    elif name == "feature_agglomeration":
        n_clusters = int(round(params["fa_n_clusters"]))
        n_clusters = min(n_clusters, X_train_shape[1])
        linkage = ["ward", "complete", "average"][int(round(params["fa_linkage"]))]
        return FeatureAgglomeration(
            n_clusters=max(2, n_clusters),
            linkage=linkage
        )
    elif name == "rbf_sampler":
        n_components = int(round(params["rbf_n_components"]))
        return RBFSampler(
            gamma=params["rbf_gamma"],
            n_components=n_components,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown preprocessor: {name}")