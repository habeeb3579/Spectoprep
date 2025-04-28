import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from typing import List, Dict, Tuple, Union, Optional
import pandas as pd
from spectoprep.pipeline.optimizer import PipelineOptimizer

class SpectroPrepPlotter:
    """
    A class for creating high-quality plots for spectroscopy data.
    
    This class provides various plotting functions specifically designed
    for spectroscopy data and pipeline optimization results.
    """
    
    @staticmethod
    def set_style(style='whitegrid', context='paper', font_scale=1.2):
        """
        Set the visual style for the plots.
        
        Parameters
        ----------
        style : str, default='whitegrid'
            The seaborn style.
        context : str, default='paper'
            The seaborn context.
        font_scale : float, default=1.2
            The font scale.
        """
        sns.set_style(style)
        sns.set_context(context, font_scale=font_scale)
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100
        
    @staticmethod
    def plot_spectra(wavenumbers: np.ndarray, 
                     spectra: np.ndarray, 
                     labels: Optional[List[str]] = None,
                     title: str = 'Spectral Data',
                     xlabel: str = 'Wavenumber (cm$^{-1}$)',
                     ylabel: str = 'Absorbance',
                     alpha: float = 0.7,
                     figsize: Tuple[int, int] = (12, 6),
                     color_map: str = 'viridis',
                     legend_loc: str = 'best',
                     grid: bool = True,
                     save_path: Optional[str] = None):
        """
        Plot spectral data.
        
        Parameters
        ----------
        wavenumbers : array-like
            The x-axis values (wavenumbers).
        spectra : array-like
            The spectra data of shape (n_samples, n_features).
        labels : list of str, optional
            Labels for each spectrum. If None, spectra are numbered.
        title : str, default='Spectral Data'
            Plot title.
        xlabel : str, default='Wavenumber (cm$^{-1}$)'
            X-axis label.
        ylabel : str, default='Absorbance'
            Y-axis label.
        alpha : float, default=0.7
            Transparency of the lines.
        figsize : tuple, default=(12, 6)
            Figure size.
        color_map : str, default='viridis'
            Colormap for the spectra.
        legend_loc : str, default='best'
            Location of the legend.
        grid : bool, default=True
            Whether to show grid.
        save_path : str, optional
            If provided, save the figure to this path.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
            
        n_spectra = spectra.shape[0]
        
        # Get colors from colormap
        cmap = plt.get_cmap(color_map)
        colors = [cmap(i/max(1, n_spectra-1)) for i in range(n_spectra)]
        
        # Plot each spectrum
        for i, spectrum in enumerate(spectra):
            label = f'Spectrum {i+1}' if labels is None else labels[i]
            ax.plot(wavenumbers, spectrum, label=label, color=colors[i], alpha=alpha)
        
        # Set labels and title
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Show grid if requested
        if grid:
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend if there are multiple spectra
        if n_spectra > 1:
            ax.legend(loc=legend_loc, frameon=True, framealpha=0.8)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    @staticmethod
    def plot_preprocessing_comparison(wavenumbers: np.ndarray,
                                    original_spectra: np.ndarray,
                                    processed_spectra: Dict[str, np.ndarray],
                                    sample_indices: Optional[List[int]] = None,
                                    figsize: Tuple[int, int] = (15, 10),
                                    title: str = 'Preprocessing Comparison',
                                    color_map: str = 'tab10',
                                    save_path: Optional[str] = None):
        """
        Plot comparison of original and processed spectra.
        
        Parameters
        ----------
        wavenumbers : array-like
            The x-axis values (wavenumbers).
        original_spectra : array-like
            The original spectra data of shape (n_samples, n_features).
        processed_spectra : dict
            Dictionary mapping preprocessing method names to processed spectra.
        sample_indices : list of int, optional
            Indices of samples to plot. If None, all samples are plotted.
        figsize : tuple, default=(15, 10)
            Figure size.
        title : str, default='Preprocessing Comparison'
            Main title for the figure.
        color_map : str, default='tab10'
            Colormap for differentiating samples.
        save_path : str, optional
            If provided, save the figure to this path.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        # Get the number of preprocessing methods
        n_methods = len(processed_spectra) + 1  # +1 for original spectra
        
        # Determine the samples to plot
        if sample_indices is None:
            sample_indices = list(range(original_spectra.shape[0]))
        n_samples = len(sample_indices)
        
        # Create a figure with subplots
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(n_methods, 1, height_ratios=[1] * n_methods)
        
        # Get colors for samples
        cmap = plt.get_cmap(color_map)
        colors = [cmap(i % 10) for i in range(n_samples)]
        
        # Plot original spectra
        ax_orig = fig.add_subplot(gs[0])
        for i, idx in enumerate(sample_indices):
            ax_orig.plot(wavenumbers, original_spectra[idx], 
                        color=colors[i], label=f'Sample {idx+1}')
        ax_orig.set_title('Original Spectra', fontsize=12)
        ax_orig.set_xlabel('')
        ax_orig.set_ylabel('Absorbance')
        ax_orig.legend(loc='best', frameon=True)
        ax_orig.grid(True, linestyle='--', alpha=0.3)
        
        # Plot processed spectra
        for i, (method_name, processed) in enumerate(processed_spectra.items(), 1):
            ax = fig.add_subplot(gs[i], sharex=ax_orig)
            for j, idx in enumerate(sample_indices):
                ax.plot(wavenumbers, processed[idx], color=colors[j], label=f'Sample {idx+1}')
            ax.set_title(f'{method_name}', fontsize=12)
            if i == n_methods - 1:
                ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=12)
            else:
                ax.set_xlabel('')
            ax.set_ylabel('Absorbance')
            ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_optimization_results(optimizer: PipelineOptimizer,
                                top_n: int = 5,
                                figsize: Tuple[int, int] = (12, 8),
                                title: str = 'Pipeline Optimization Results',
                                save_path: Optional[str] = None):
        """
        Plot optimization results from PipelineOptimizer.
        
        Parameters
        ----------
        optimizer : PipelineOptimizer
            The fitted pipeline optimizer.
        top_n : int, default=5
            Number of top pipelines to display.
        figsize : tuple, default=(12, 8)
            Figure size.
        title : str, default='Pipeline Optimization Results'
            Plot title.
        save_path : str, optional
            If provided, save the figure to this path.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        if optimizer.optimizer is None:
            raise ValueError("No optimization results found. Run bayesian_optimize() first.")
        
        # Get all results
        results = optimizer.get_all_tested_pipelines()
        
        # Sort by RMSE (ascending)
        sorted_results = sorted(results, key=lambda x: x['rmse'])
        
        # Select top N results
        top_results = sorted_results[:top_n]
        
        # Extract data for plotting
        pipelines = [' → '.join(res['pipeline_config']) for res in top_results]
        rmses = [res['rmse'] for res in top_results]
        r2s = [res['r2'] for res in top_results]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Plot RMSE values
        ax1.barh(pipelines, rmses, color='skyblue', alpha=0.8)
        ax1.set_title('RMSE (lower is better)', fontsize=12)
        ax1.set_ylabel('Pipeline Configuration')
        ax1.grid(True, linestyle='--', alpha=0.3, axis='x')
        
        # Add RMSE values as text
        for i, rmse in enumerate(rmses):
            ax1.text(rmse + max(rmses) * 0.01, i, f'{rmse:.4f}', 
                    va='center', fontsize=10)
        
        # Plot R² values
        ax2.barh(pipelines, r2s, color='lightgreen', alpha=0.8)
        ax2.set_title('R² (higher is better)', fontsize=12)
        ax2.set_ylabel('Pipeline Configuration')
        ax2.set_xlabel('Score')
        ax2.grid(True, linestyle='--', alpha=0.3, axis='x')
        
        # Add R² values as text
        for i, r2 in enumerate(r2s):
            ax2.text(r2 + max(r2s) * 0.01, i, f'{r2:.4f}', 
                    va='center', fontsize=10)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_prediction_scatter(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               title: str = 'Prediction Performance',
                               xlabel: str = 'Measured',
                               ylabel: str = 'Predicted',
                               figsize: Tuple[int, int] = (10, 8),
                               alpha: float = 0.7,
                               color: str = 'blue',
                               add_metrics: bool = True,
                               save_path: Optional[str] = None):
        """
        Create a scatter plot of predicted vs true values.
        
        Parameters
        ----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted target values.
        title : str, default='Prediction Performance'
            Plot title.
        xlabel : str, default='Measured'
            X-axis label.
        ylabel : str, default='Predicted'
            Y-axis label.
        figsize : tuple, default=(10, 8)
            Figure size.
        alpha : float, default=0.7
            Transparency of the points.
        color : str, default='blue'
            Color of the scatter points.
        add_metrics : bool, default=True
            Whether to add RMSE and R² metrics to the plot.
        save_path : str, optional
            If provided, save the figure to this path.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot scatter points
        ax.scatter(y_true, y_pred, alpha=alpha, color=color, edgecolor='k', linewidth=0.5)
        
        # Calculate and plot identity line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        padding = (max_val - min_val) * 0.05
        line_x = np.array([min_val - padding, max_val + padding])
        ax.plot(line_x, line_x, 'k--', alpha=0.7, label='Identity Line')
        
        # Set axis limits
        ax.set_xlim(min_val - padding, max_val + padding)
        ax.set_ylim(min_val - padding, max_val + padding)
        
        # Add metrics text if requested
        if add_metrics:
            metrics_text = f'RMSE = {rmse:.4f}\nR² = {r2:.4f}'
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
                   verticalalignment='top', fontsize=12)
        
        # Set labels and title
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    @staticmethod
    def plot_optimization_progress(optimizer: PipelineOptimizer,
                                 figsize: Tuple[int, int] = (12, 6),
                                 title: str = 'Optimization Progress',
                                 save_path: Optional[str] = None):
        """
        Plot optimization progress over iterations.
        
        Parameters
        ----------
        optimizer : PipelineOptimizer
            The fitted pipeline optimizer.
        figsize : tuple, default=(12, 6)
            Figure size.
        title : str, default='Optimization Progress'
            Plot title.
        save_path : str, optional
            If provided, save the figure to this path.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        if optimizer.optimizer is None:
            raise ValueError("No optimization results found. Run bayesian_optimize() first.")
        
        # Get all results
        results = optimizer.get_all_tested_pipelines()
        
        # Extract iteration and RMSE data
        iterations = [res['trial'] for res in results if res['trial'] is not None]
        rmses = [res['rmse'] for res in results if res['trial'] is not None]
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame({'Iteration': iterations, 'RMSE': rmses})
        df = df.sort_values('Iteration')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot RMSE values
        ax.plot(df['Iteration'], df['RMSE'], marker='o', linestyle='-', color='blue', alpha=0.7)
        
        # Calculate and plot running minimum RMSE
        running_min = df['RMSE'].cummin()
        ax.plot(df['Iteration'], running_min, marker='', linestyle='--', color='red', 
               alpha=0.8, label='Best RMSE')
        
        # Add labels and title
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    @staticmethod
    def plot_feature_importance(wavenumbers: np.ndarray,
                              coefficients: np.ndarray,
                              title: str = 'Feature Importance',
                              xlabel: str = 'Wavenumber (cm$^{-1}$)',
                              ylabel: str = 'Coefficient Value',
                              figsize: Tuple[int, int] = (12, 6),
                              color: str = 'purple',
                              highlight_threshold: Optional[float] = None,
                              highlight_color: str = 'red',
                              save_path: Optional[str] = None):
        """
        Plot feature importance from model coefficients.
        
        Parameters
        ----------
        wavenumbers : array-like
            The x-axis values (wavenumbers).
        coefficients : array-like
            Model coefficients corresponding to each wavenumber.
        title : str, default='Feature Importance'
            Plot title.
        xlabel : str, default='Wavenumber (cm$^{-1}$)'
            X-axis label.
        ylabel : str, default='Coefficient Value'
            Y-axis label.
        figsize : tuple, default=(12, 6)
            Figure size.
        color : str, default='purple'
            Color of the line.
        highlight_threshold : float, optional
            If provided, highlights coefficients with absolute values above this threshold.
        highlight_color : str, default='red'
            Color for highlighted coefficients.
        save_path : str, optional
            If provided, save the figure to this path.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot coefficients
        ax.plot(wavenumbers, coefficients, color=color, alpha=0.7)
        
        # Highlight important features if threshold is provided
        if highlight_threshold is not None:
            important_mask = np.abs(coefficients) > highlight_threshold
            if np.any(important_mask):
                ax.scatter(wavenumbers[important_mask], coefficients[important_mask], 
                         color=highlight_color, s=50, zorder=3, 
                         label=f'|Coef| > {highlight_threshold}')
                ax.legend(loc='best')
        
        # Add zero line
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Set labels and title
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax