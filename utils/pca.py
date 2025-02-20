import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def setup_logging() -> logging.Logger:
    """Configure logging for the analysis."""
    logger = logging.getLogger('pca_analysis')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def preprocess_data(data: np.ndarray, logger: logging.Logger) -> np.ndarray:
    """
    Preprocess the EEG data by standardizing and handling missing values.
    
    Args:
        data: Raw EEG data array of shape (n_channels, n_samples)
        logger: Logger instance
    
    Returns:
        Standardized data array
    """
    logger.info("Starting data preprocessing...")
    
    # Handle missing and infinite values
    if np.any(np.isnan(data)):
        logger.warning("NaN values found in data. Replacing with zeros.")
        data = np.nan_to_num(data)
    
    if np.any(np.isinf(data)):
        logger.warning("Infinite values found in data. Replacing with zeros.")
        data = np.nan_to_num(data, posinf=0, neginf=0)
    
    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data.T).T
    
    logger.info("Data preprocessing completed.")
    return standardized_data

def compute_pca(data: np.ndarray, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PCA on the preprocessed data.
    
    Args:
        data: Preprocessed data array
        logger: Logger instance
    
    Returns:
        Tuple containing eigenvalues, eigenvectors, explained variance ratio, and cumulative variance
    """
    logger.info("Computing PCA...")
    
    # Compute covariance matrix
    covariance_matrix = np.cov(data)
    
    # Compute eigenvalues and eigenvectors using eigh for symmetric matrices
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Calculate variance ratios
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues) * 100
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    logger.info("PCA computation completed.")
    return eigenvalues, eigenvectors, explained_variance_ratio, cumulative_variance

def find_optimal_components(cumulative_variance: np.ndarray, 
                          threshold: float = 98.0,
                          logger: logging.Logger = None) -> int:
    """
    Find optimal number of components based on explained variance threshold.
    
    Args:
        cumulative_variance: Array of cumulative explained variance
        threshold: Minimum cumulative variance to explain (default: 95%)
        logger: Logger instance
    
    Returns:
        Optimal number of components
    """
    optimal_components = np.argmax(cumulative_variance >= threshold) + 1
    if logger:
        logger.info(f"Optimal components for {threshold}% variance: {optimal_components}")
    return optimal_components

def transform_data(data: np.ndarray, 
                  eigenvectors: np.ndarray, 
                  n_components: int) -> np.ndarray:
    """
    Transform data using selected principal components.
    
    Args:
        data: Preprocessed data array
        eigenvectors: Matrix of eigenvectors
        n_components: Number of components to use
        
    Returns:
        Transformed data array
    """
    return np.dot(eigenvectors[:, :n_components].T, data)

def create_components_dataframe(eigenvalues: np.ndarray, 
                              explained_variance_ratio: np.ndarray,
                              cumulative_variance: np.ndarray) -> pd.DataFrame:
    """
    Create a DataFrame with PCA component information.
    
    Args:
        eigenvalues: Array of eigenvalues
        explained_variance_ratio: Array of explained variance ratios
        cumulative_variance: Array of cumulative variance
        
    Returns:
        DataFrame with component information
    """
    return pd.DataFrame({
        'Component': range(1, len(eigenvalues) + 1),
        'Eigenvalue': eigenvalues,
        'Explained_Variance(%)': explained_variance_ratio,
        'Cumulative_Variance(%)': cumulative_variance
    })

def plot_pca_results(explained_variance_ratio: np.ndarray,
                    cumulative_variance: np.ndarray,
                    transformed_data: np.ndarray,
                    eigenvectors: np.ndarray,
                    max_components: Optional[int] = None) -> None:
    """
    Create visualization plots for PCA results.
    
    Args:
        explained_variance_ratio: Array of explained variance ratios
        cumulative_variance: Array of cumulative variance
        transformed_data: Transformed data array
        eigenvectors: Matrix of eigenvectors
        max_components: Maximum number of components to plot
    """
    if max_components is None:
        max_components = min(10, len(explained_variance_ratio))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Scree plot
    axes[0, 0].plot(range(1, max_components + 1), 
                    explained_variance_ratio[:max_components],
                    'bo-')
    axes[0, 0].set_title('Scree Plot')
    axes[0, 0].set_xlabel('Principal Component')
    axes[0, 0].set_ylabel('Explained Variance Ratio (%)')
    axes[0, 0].grid(True)
    
    # Cumulative variance plot
    axes[0, 1].plot(range(1, max_components + 1),
                    cumulative_variance[:max_components],
                    'ro-')
    axes[0, 1].set_title('Cumulative Explained Variance')
    axes[0, 1].set_xlabel('Number of Components')
    axes[0, 1].set_ylabel('Cumulative Explained Variance (%)')
    axes[0, 1].grid(True)
    
    # First two principal components
    if transformed_data.shape[0] >= 2:
        axes[1, 0].scatter(transformed_data[0, :],
                          transformed_data[1, :],
                          alpha=0.5)
        axes[1, 0].set_title('First Two Principal Components')
        axes[1, 0].set_xlabel('PC1')
        axes[1, 0].set_ylabel('PC2')
        axes[1, 0].grid(True)
    
    # Component weights heatmap
    if max_components <= 25:
        im = axes[1, 1].imshow(
            eigenvectors[:, :max_components],
            aspect='auto',
            cmap='coolwarm'
        )
        axes[1, 1].set_title('Component Weights Heatmap')
        axes[1, 1].set_xlabel('Principal Component')
        axes[1, 1].set_ylabel('Original Feature')
        plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()

def analyze_eeg_pca(raw_data: np.ndarray, variance_threshold: float = 98.0) -> Dict:
    """
    Main function to perform PCA analysis on EEG data.
    
    Args:
        raw_data: Raw EEG data array
        variance_threshold: Threshold for explained variance
    
    Returns:
        Dictionary containing analysis results
    """
    # Setup logging
    logger = setup_logging()
    
    # Preprocess data
    standardized_data = preprocess_data(raw_data, logger)
    
    # Compute PCA
    eigenvalues, eigenvectors, explained_variance_ratio, cumulative_variance = compute_pca(
        standardized_data, logger
    )
    
    # Find optimal number of components
    n_components = find_optimal_components(
        cumulative_variance, 
        threshold=variance_threshold,
        logger=logger
    )
    
    # Transform data
    transformed_data = transform_data(
        standardized_data, 
        eigenvectors, 
        n_components
    )
    
    # Create components DataFrame
    components_df = create_components_dataframe(
        eigenvalues,
        explained_variance_ratio,
        cumulative_variance
    )
    
    # Plot results
    plot_pca_results(
        explained_variance_ratio,
        cumulative_variance,
        transformed_data,
        eigenvectors
    )
    
    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'transformed_data': transformed_data,
        'components_df': components_df,
        'n_components': n_components
    }