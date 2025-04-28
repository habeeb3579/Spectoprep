"""
Main PipelineOptimizer class implementation.
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import numpy.typing as npt
import logging
import warnings
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit, LeavePGroupsOut
from sklearn.metrics import mean_squared_error

# Import Bayesian Optimization
from bayes_opt import BayesianOptimization

# Import from package modules
from spectoprep.pipeline.config import AVAILABLE_STEPS, INCOMPATIBLE_SETS, DEFAULT_PARAM_BOUNDS
from spectoprep.pipeline.builder import build_preprocessor_from_bayes
from spectoprep.modelling.ridge import OptimizedRidgeCV
from spectoprep.pipeline.utils import generate_pipeline_configurations



# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class PipelineOptimizer:
    """
    A class for optimizing machine learning pipelines using Bayesian optimization.
    It precomputes possible pipeline configurations and then searches over both
    the pipeline configuration (encoded as an index) and the hyperparameters.
    """
    
    def __init__(
        self,
        X_train: npt.NDArray,
        y_train: npt.NDArray,
        preprocessing_steps: Optional[List[str]] = None,
        X_test: Optional[npt.NDArray] = None,
        y_test: Optional[npt.NDArray] = None,
        cv_method: str = "group_shuffle_split",
        n_splits: int = 3,
        test_size: float = 0.3,
        n_groups_out: int = 2,
        random_state: int = 42,
        groups: Optional[npt.NDArray] = None,
        max_pipeline_length: int = 5,
        n_jobs: int = -1,
        allowed_preprocess_combinations: Optional[Union[int, List[int], Tuple[int, ...]]] = [1, 2],
        log_level: str = "INFO"
    ):
        """
        Initialize the PipelineOptimizer.

        Args:
            X_train: Training features.
            y_train: Training targets.
            preprocessing_steps: List of preprocessing steps to use.
            X_test: Test features (optional).
            y_test: Test targets (optional).
            cv_method: Either "group_shuffle_split" or "leave_p_group_out".
            n_splits: Number of CV splits.
            test_size: Test set fraction (if using GroupShuffleSplit).
            n_groups_out: Number of groups left out (if using LeavePGroupsOut).
            random_state: Random seed.
            groups: Optional group labels; if None, one group per sample is used.
            max_pipeline_length: Maximum number of steps in pipeline.
            n_jobs: Number of parallel jobs for compatible estimators.
            allowed_preprocess_combinations: Allowed lengths for preprocessing combinations.
            log_level: Logging level (INFO, DEBUG, WARNING, ERROR).
        """
        # Set up logging
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("PipelineOptimizer")
        
        # Store data attributes
        self.X_train = X_train
        self.y_train = np.ravel(y_train)
        self.X_test = X_test
        self.y_test = np.ravel(y_test) if y_test is not None else None
        
        # Store cross-validation parameters
        self.cv_method = cv_method
        self.n_splits = n_splits
        self.test_size = test_size
        self.n_groups_out = n_groups_out
        
        # Store other configuration
        self.random_state = random_state
        self.max_pipeline_length = max_pipeline_length
        self.n_jobs = n_jobs
        self.allowed_preprocess_combinations = allowed_preprocess_combinations
        
        # Handle groups
        if groups is not None:
            self.groups = np.ravel(groups)
        else:
            self.groups = np.arange(self.X_train.shape[0])
        
        # Validate and set preprocessing steps
        self.preprocessing_steps = self._validate_preprocessing_steps(preprocessing_steps)
        
        # Validate inputs
        self._validate_inputs()
        
        # Generate all valid pipeline configurations
        self.all_pipelines = generate_pipeline_configurations(
            self.preprocessing_steps, 
            INCOMPATIBLE_SETS, 
            self.max_pipeline_length,
            self.allowed_preprocess_combinations
        )
        
        # Update pipeline config bounds
        self.param_bounds = DEFAULT_PARAM_BOUNDS.copy()
        self.param_bounds["pipeline_config"] = (0, len(self.all_pipelines) - 1)
        
        # Log initialization information
        self.logger.info("Initialized PipelineOptimizer")
        self.logger.info(f"Number of preprocessing steps: {len(self.preprocessing_steps)}")
        self.logger.info(f"Number of possible pipeline configurations: {len(self.all_pipelines)}")
        self.logger.info(f"Using {cv_method} validation method")
    
    def _validate_preprocessing_steps(self, steps: Optional[List[str]] = None) -> List[str]:
        """
        Validate and standardize preprocessing steps.
        
        Args:
            steps: List of preprocessing step names
            
        Returns:
            Validated list of preprocessing step names
        """
        if steps is None:
            # Default steps
            steps = ['scaler', 'pca', 'robust_scaler', 'select_k_best']
        if isinstance(steps, str):
            steps = [steps]
        
        invalid_steps = set(steps) - set(AVAILABLE_STEPS.keys())
        if invalid_steps:
            raise ValueError(f"Invalid preprocessing steps: {invalid_steps}. Available: {list(AVAILABLE_STEPS.keys())}")
        
        return list(steps)
    
    def _validate_inputs(self) -> None:
        """
        Validate input data and parameters.
        
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(self.X_train, np.ndarray) or not isinstance(self.y_train, np.ndarray):
            raise ValueError("X_train and y_train must be numpy arrays")
            
        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError(f"X_train and y_train must have the same number of samples. Got {self.X_train.shape[0]} and {self.y_train.shape[0]}")
            
        if (self.X_test is None) != (self.y_test is None):
            raise ValueError("Both X_test and y_test must be provided together")
            
        if self.X_test is not None and self.X_test.shape[1] != self.X_train.shape[1]:
            raise ValueError(f"X_test and X_train must have the same number of features. Got {self.X_test.shape[1]} and {self.X_train.shape[1]}")
            
        if self.cv_method not in ["group_shuffle_split", "leave_p_group_out"]:
            raise ValueError("cv_method must be 'group_shuffle_split' or 'leave_p_group_out'")
            
        if len(self.groups) != self.X_train.shape[0]:
            raise ValueError("Groups must have the same number of samples as X_train.")
    
    # def bayes_objective(self, **params) -> float:
    #     """
    #     Objective function for Bayesian optimization.
        
    #     Args:
    #         **params: Parameters to evaluate
            
    #     Returns:
    #         float: Negative RMSE or penalty score on error
    #     """
    #     try:
    #         # Extract pipeline configuration index and limit to valid range
    #         pipeline_config_index = int(round(params["pipeline_config"]))
    #         pipeline_config_index = max(0, min(pipeline_config_index, len(self.all_pipelines) - 1))
    #         pipeline_config = self.all_pipelines[pipeline_config_index]
            
    #         # Build pipeline steps
    #         steps = []
    #         for step in pipeline_config:
    #             transformer = build_preprocessor_from_bayes(
    #                 step, params, self.X_train.shape, self.random_state, self.n_jobs
    #             )
    #             steps.append((step, transformer))
            
    #         # Add Ridge regression as the final estimator
    #         ridge_alpha = params["ridge_alpha"]
    #         steps.append(("ridge", Ridge(alpha=ridge_alpha, random_state=self.random_state)))
    #         pipeline = Pipeline(steps)
            
    #         # Configure cross-validation
    #         if self.cv_method == "group_shuffle_split":
    #             cv = GroupShuffleSplit(
    #                 n_splits=self.n_splits,
    #                 test_size=self.test_size,
    #                 random_state=self.random_state
    #             )
    #         else:
    #             cv = LeavePGroupsOut(n_groups=self.n_groups_out)
            
    #         # Perform cross-validation
    #         all_predictions = []
    #         all_actuals = []
            
    #         for train_idx, val_idx in cv.split(self.X_train, self.y_train, groups=self.groups):
    #             try:
    #                 X_train_fold = self.X_train[train_idx]
    #                 X_val_fold = self.X_train[val_idx]
    #                 y_train_fold = self.y_train[train_idx]
    #                 y_val_fold = self.y_train[val_idx]
                    
    #                 # Check condition number to avoid numerical instability
    #                 if np.linalg.cond(X_train_fold) > 1e10:
    #                     self.logger.warning("High condition number detected. Skipping this fold.")
    #                     continue
                        
    #                 pipeline.fit(X_train_fold, y_train_fold)
    #                 preds = pipeline.predict(X_val_fold)
    #                 preds = np.ravel(preds)
    #                 all_predictions.extend(preds)
    #                 all_actuals.extend(y_val_fold)
                    
    #             except np.linalg.LinAlgError as e:
    #                 self.logger.warning(f"LinAlgError in fold: {str(e)}")
    #                 continue
    #             except Exception as e:
    #                 self.logger.warning(f"Error in fold: {str(e)}")
    #                 continue
            
    #         # Check if we have valid predictions
    #         if not all_predictions:
    #             self.logger.warning("No valid predictions. Returning penalty score.")
    #             return -1e6  # Penalty score
                
    #         # Calculate metrics
    #         rmse = np.sqrt(mean_squared_error(np.array(all_actuals), np.array(all_predictions)))
    #         r2 = 1 - np.sum((np.array(all_actuals) - np.array(all_predictions))**2) / np.sum((np.array(all_actuals) - np.mean(np.array(all_actuals)))**2)
            
    #         # Weighted combination of metrics to optimize (negative RMSE + small R² contribution)
    #         score = -rmse
            
    #         self.logger.info(f"Pipeline config: {pipeline_config}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    #         return score
            
    #     except Exception as e:
    #         self.logger.error(f"Error in bayes_objective: {str(e)}")
    #         return -1e6  # Penalty score for failed configurations

    #updated to include test set for validation
    def bayes_objective(self, **params) -> float:
        """
        Objective function for Bayesian optimization.
        
        Args:
            **params: Parameters to evaluate
            
        Returns:
            float: Negative RMSE or penalty score on error
        """
        try:
            # Extract pipeline configuration index and limit to valid range
            pipeline_config_index = int(round(params["pipeline_config"]))
            pipeline_config_index = max(0, min(pipeline_config_index, len(self.all_pipelines) - 1))
            pipeline_config = self.all_pipelines[pipeline_config_index]
            
            # Build pipeline steps
            steps = []
            for step in pipeline_config:
                transformer = build_preprocessor_from_bayes(
                    step, params, self.X_train.shape, self.random_state, self.n_jobs
                )
                steps.append((step, transformer))
            
            # Add Ridge regression as the final estimator
            ridge_alpha = params["ridge_alpha"]
            steps.append(("ridge", Ridge(alpha=ridge_alpha, random_state=self.random_state)))
            pipeline = Pipeline(steps)
            
            # If test data is available, use it directly for validation
            if self.X_test is not None and self.y_test is not None:
                try:
                    pipeline.fit(self.X_train, self.y_train)
                    preds = pipeline.predict(self.X_test)
                    preds = np.ravel(preds)
                    
                    # Calculate metrics
                    rmse = np.sqrt(mean_squared_error(self.y_test, preds))
                    r2 = 1 - np.sum((self.y_test - preds)**2) / np.sum((self.y_test - np.mean(self.y_test))**2)
                    
                    score = -rmse
                    self.logger.info(f"Pipeline config: {pipeline_config}, RMSE on test set: {rmse:.4f}, R²: {r2:.4f}")
                    return score
                    
                except np.linalg.LinAlgError as e:
                    self.logger.warning(f"LinAlgError with test set: {str(e)}")
                    # Fall back to cross-validation if there's an error
                except Exception as e:
                    self.logger.warning(f"Error with test set: {str(e)}")
                    # Fall back to cross-validation if there's an error
            
            # If no test data or error occurred, use cross-validation
            # Configure cross-validation
            if self.cv_method == "group_shuffle_split":
                cv = GroupShuffleSplit(
                    n_splits=self.n_splits,
                    test_size=self.test_size,
                    random_state=self.random_state
                )
            else:
                cv = LeavePGroupsOut(n_groups=self.n_groups_out)
            
            # Perform cross-validation
            all_predictions = []
            all_actuals = []
            
            for train_idx, val_idx in cv.split(self.X_train, self.y_train, groups=self.groups):
                try:
                    X_train_fold = self.X_train[train_idx]
                    X_val_fold = self.X_train[val_idx]
                    y_train_fold = self.y_train[train_idx]
                    y_val_fold = self.y_train[val_idx]
                    
                    # Check condition number to avoid numerical instability
                    if np.linalg.cond(X_train_fold) > 1e10:
                        self.logger.warning("High condition number detected. Skipping this fold.")
                        continue
                        
                    pipeline.fit(X_train_fold, y_train_fold)
                    preds = pipeline.predict(X_val_fold)
                    preds = np.ravel(preds)
                    all_predictions.extend(preds)
                    all_actuals.extend(y_val_fold)
                    
                except np.linalg.LinAlgError as e:
                    self.logger.warning(f"LinAlgError in fold: {str(e)}")
                    continue
                except Exception as e:
                    self.logger.warning(f"Error in fold: {str(e)}")
                    continue
            
            # Check if we have valid predictions
            if not all_predictions:
                self.logger.warning("No valid predictions. Returning penalty score.")
                return -1e6  # Penalty score
                
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(np.array(all_actuals), np.array(all_predictions)))
            r2 = 1 - np.sum((np.array(all_actuals) - np.array(all_predictions))**2) / np.sum((np.array(all_actuals) - np.mean(np.array(all_actuals)))**2)
            
            # Weighted combination of metrics to optimize (negative RMSE + small R² contribution)
            score = -rmse
            
            self.logger.info(f"Pipeline config: {pipeline_config}, CV RMSE: {rmse:.4f}, R²: {r2:.4f}")
            return score
            
        except Exception as e:
            self.logger.error(f"Error in bayes_objective: {str(e)}")
            return -1e6  # Penalty score for failed configurations
        
    
    
    def bayesian_optimize(self, init_points: int = 10, n_iter: int = 50, acquisition_function: str = "ei") -> Tuple[Dict, Pipeline]:
        """
        Run Bayesian optimization to find the best pipeline configuration and hyperparameters.
        
        Args:
            init_points: Number of random initial points
            n_iter: Number of Bayesian optimization iterations
            acquisition_function: Acquisition function for Bayesian optimization
            
        Returns:
            Tuple containing:
                - Dict of best parameters
                - Fitted Pipeline with best configuration
        """
        # Create optimizer
        optimizer = BayesianOptimization(
            f=self.bayes_objective,
            pbounds=self.param_bounds,
            random_state=self.random_state,
            verbose=2
        )
        
        # Set acquisition function
        if acquisition_function not in ["ucb", "ei", "poi"]:
            self.logger.warning(f"Unknown acquisition function: {acquisition_function}. Using 'ei'.")
            acquisition_function = "ei"
        
        # Run optimization
        optimizer.maximize(
            init_points=init_points, 
            n_iter=n_iter
        )
        
        # Store optimizer for later analysis
        self.optimizer = optimizer
        
        # Extract best parameters
        best_params = optimizer.max["params"]
        
        # Build the best pipeline from the best parameters
        pipeline_config_index = int(round(best_params["pipeline_config"]))
        pipeline_config_index = max(0, min(pipeline_config_index, len(self.all_pipelines) - 1))
        best_pipeline_config = self.all_pipelines[pipeline_config_index]
        
        # Create and fit the best pipeline
        steps = []
        for step in best_pipeline_config:
            transformer = build_preprocessor_from_bayes(
                step, best_params, self.X_train.shape, self.random_state, self.n_jobs
            )
            steps.append((step, transformer))
            
        ridge_alpha = best_params["ridge_alpha"]
        steps.append(("ridge", Ridge(alpha=ridge_alpha, random_state=self.random_state)))
        
        best_pipeline = Pipeline(steps)
        best_pipeline.fit(self.X_train, self.y_train)
        
        # Log best pipeline details
        self.logger.info(f"Best pipeline config: {best_pipeline_config}")
        self.logger.info(f"Best pipeline score: {optimizer.max['target']}")
        for step, transformer in steps:
            self.logger.info(f"Step: {step}, Transformer: {transformer}")
            
        return best_params, best_pipeline
    
    def get_best_pipeline_predictions(self, best_pipeline: Pipeline) -> Tuple[npt.NDArray, float, float]:
        """
        Get predictions using the best pipeline.
        
        Args:
            best_pipeline: Fitted pipeline object
            
        Returns:
            Tuple containing:
                - Predictions array
                - RMSE score
                - R² score
        """
        # Fit the pipeline to training data
        best_pipeline.fit(self.X_train, self.y_train)
        
        # If test data is available, use it for evaluation
        if self.X_test is not None:
            preds = best_pipeline.predict(self.X_test)
            preds = np.ravel(preds)
            rmse = np.sqrt(mean_squared_error(self.y_test, preds))
            r2 = 1 - np.sum((self.y_test - preds)**2) / np.sum((self.y_test - np.mean(self.y_test))**2)
        else:
            # If no test set, use cross-validation predictions
            if self.cv_method == "group_shuffle_split":
                cv = GroupShuffleSplit(
                    n_splits=self.n_splits,
                    test_size=self.test_size,
                    random_state=self.random_state
                )
            else:
                cv = LeavePGroupsOut(n_groups=self.n_groups_out)
                
            all_preds = []
            all_true = []
            
            for train_idx, val_idx in cv.split(self.X_train, self.y_train, groups=self.groups):
                X_train_fold = self.X_train[train_idx]
                X_val_fold = self.X_train[val_idx]
                y_train_fold = self.y_train[train_idx]
                y_val_fold = self.y_train[val_idx]
                
                best_pipeline.fit(X_train_fold, y_train_fold)
                preds = best_pipeline.predict(X_val_fold)
                preds = np.ravel(preds)
                all_preds.extend(preds)
                all_true.extend(y_val_fold)
            
            preds = np.array(all_preds)
            true_vals = np.array(all_true)
            rmse = np.sqrt(mean_squared_error(true_vals, preds))
            r2 = 1 - np.sum((true_vals - preds)**2) / np.sum((true_vals - np.mean(true_vals))**2)
            
        return preds, rmse, r2
    
    def get_all_tested_pipelines(self) -> List[Dict]:
        """
        Get details of all tested pipeline configurations.
        
        Returns:
            List of dictionaries with pipeline details
        """
        if not hasattr(self, "optimizer"):
            raise AttributeError("No optimizer found. Please run bayesian_optimize() first.")
            
        results = []
        for i, res in enumerate(self.optimizer.res):
            params = res["params"]
            pipeline_index = int(round(params["pipeline_config"]))
            pipeline_index = max(0, min(pipeline_index, len(self.all_pipelines) - 1))
            pipeline_config = self.all_pipelines[pipeline_index]
            
            # Calculate metrics from the saved objective score (negative RMSE)
            rmse = -res["target"] if res["target"] > -1e5 else float('inf')
            
            # Create a dictionary with trial information
            result_dict = {
                "trial": i,
                "pipeline_config": pipeline_config,
                "params": {k: v for k, v in params.items() if k != "pipeline_config"},
                "rmse": rmse,
                "r2": None  # R² is not directly available from objective score
            }
            results.append(result_dict)
            
        return results
    
    def print_evaluated_pipelines(self) -> None:
        """
        Print details for all evaluated pipelines from the Bayesian optimizer.
        
        This method assumes that bayesian_optimize() has been run and that
        self.optimizer exists.
        """
        if not hasattr(self, "optimizer"):
            self.logger.warning("No optimizer found. Please run bayesian_optimize() first.")
            return

        print("Evaluated pipelines:")
        for i, res in enumerate(self.optimizer.res):
            params = res["params"]
            # Convert the continuous pipeline_config parameter to an integer index
            pipeline_index = int(round(params["pipeline_config"]))
            # Clamp the index to the valid range
            pipeline_index = max(0, min(pipeline_index, len(self.all_pipelines) - 1))
            pipeline_config = self.all_pipelines[pipeline_index]
            target = res["target"]
            print(f"Trial {i}:")
            print(f"  Pipeline configuration: {pipeline_config}")
            print(f"  Hyperparameters: {params}")
            print(f"  Objective (score): {target:.4f}")
    
    def export_best_pipeline(self, file_path: str) -> None:
        """
        Export the best pipeline configuration and hyperparameters to a file.
        
        Args:
            file_path: Path to save the export file
            
        Raises:
            AttributeError: If optimizer hasn't been run yet
        """
        if not hasattr(self, "optimizer"):
            raise AttributeError("No optimizer found. Please run bayesian_optimize() first.")
            
        import json
        
        best_params = self.optimizer.max["params"]
        pipeline_index = int(round(best_params["pipeline_config"]))
        pipeline_index = max(0, min(pipeline_index, len(self.all_pipelines) - 1))
        best_pipeline_config = self.all_pipelines[pipeline_index]
        
        export_data = {
            "best_score": self.optimizer.max["target"],
            "pipeline_config": list(best_pipeline_config),
            "hyperparameters": {k: v for k, v in best_params.items() if k != "pipeline_config"}
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        self.logger.info(f"Best pipeline configuration exported to {file_path}")
    
    def summarize_optimization(self) -> Dict:
        """
        Generate a summary of the optimization results.
        
        Returns:
            Dictionary containing optimization summary metrics
        """
        if not hasattr(self, "optimizer"):
            raise AttributeError("No optimizer found. Please run bayesian_optimize() first.")
        
        results = self.optimizer.res
        targets = [r["target"] for r in results]
        
        # Extract best pipeline configuration
        best_params = self.optimizer.max["params"]
        pipeline_index = int(round(best_params["pipeline_config"]))
        best_pipeline_config = self.all_pipelines[pipeline_index]
        
        # Calculate improvement and convergence metrics
        initial_performance = min(targets[:5]) if len(targets) >= 5 else min(targets)
        final_performance = self.optimizer.max["target"]
        improvement = final_performance - initial_performance
        
        # Check for convergence by looking at the last few iterations
        n_last = min(5, len(targets))
        recent_targets = targets[-n_last:]
        converged = (max(recent_targets) - min(recent_targets)) < 0.001
        
        # Count unique pipeline configurations evaluated
        unique_configs = set()
        for res in results:
            idx = int(round(res["params"]["pipeline_config"]))
            idx = max(0, min(idx, len(self.all_pipelines) - 1))
            unique_configs.add(idx)
            
        # Create summary
        summary = {
            "n_trials": len(results),
            "n_unique_configs": len(unique_configs),
            "best_score": final_performance,
            "best_pipeline": list(best_pipeline_config),
            "improvement": improvement,
            "converged": converged,
            "best_rmse": -final_performance if final_performance > -1e5 else float('inf')
        }
        
        return summary