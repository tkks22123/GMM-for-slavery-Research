import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from sklearn.datasets import make_blobs
import seaborn as sns
import time
import logging
from typing import Tuple, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('GMM-EM')

class GaussianMixtureModel:
    """
    Gaussian Mixture Model with Expectation-Maximization algorithm
    
    Attributes:
        n_components (int): Number of Gaussian components
        max_iter (int): Maximum iterations for EM algorithm
        tol (float): Convergence tolerance
        weights (np.ndarray): Component weights
        means (np.ndarray): Component means
        covariances (np.ndarray): Component covariance matrices
        log_likelihood_history (list): Log-likelihood at each iteration
    """
    
    def __init__(self, n_components: int = 3, max_iter: int = 100, tol: float = 1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.means = None
        self.covariances = None
        self.log_likelihood_history = []
        self.converged = False
        logger.info(f"Initialized GMM with {n_components} components")
    
    def _initialize_parameters(self, X: np.ndarray):
        """Initialize model parameters using K-means++ strategy"""
        n_samples, n_features = X.shape
        
        # Initialize weights uniformly
        self.weights = np.ones(self.n_components) / self.n_components
        
        # Initialize means using K-means++ centroids
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        
        # Initialize covariances as identity matrices
        self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])
        
        logger.debug("Parameters initialized")
    
    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """Expectation step: compute posterior probabilities"""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            # Calculate multivariate normal probability density
            responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(
                X, mean=self.means[k], cov=self.covariances[k]
            )
        
        # Normalize responsibilities
        sum_resp = responsibilities.sum(axis=1, keepdims=True)
        responsibilities /= sum_resp
        
        # Handle numerical stability
        responsibilities = np.clip(responsibilities, 1e-15, 1-1e-15)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        
        return responsibilities
    
    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """Maximization step: update model parameters"""
        n_samples, n_features = X.shape
        
        # Effective number of data points assigned to each component
        Nk = responsibilities.sum(axis=0)
        
        # Update weights
        self.weights = Nk / n_samples
        
        # Update means
        self.means = np.zeros((self.n_components, n_features))
        for k in range(self.n_components):
            self.means[k] = (responsibilities[:, k] @ X) / Nk[k]
        
        # Update covariances
        self.covariances = np.zeros((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = (responsibilities[:, k, None, None] * 
                                  np.einsum('ij,ik->ijk', diff, diff)).sum(axis=0) / Nk[k]
            
            # Add regularization for numerical stability
            self.covariances[k] += 1e-6 * np.eye(n_features)
    
    def _compute_log_likelihood(self, X: np.ndarray) -> float:
        """Compute log-likelihood of data under current model"""
        log_likelihood = 0
        for k in range(self.n_components):
            log_prob = multivariate_normal.logpdf(
                X, mean=self.means[k], cov=self.covariances[k]
            )
            log_likelihood += np.sum(self.weights[k] * np.exp(log_prob))
        
        # Use log-sum-exp for numerical stability
        weighted_log_probs = []
        for k in range(self.n_components):
            log_prob = multivariate_normal.logpdf(X, self.means[k], self.covariances[k])
            weighted_log_probs.append(np.log(self.weights[k]) + log_prob)
        
        log_likelihood = np.sum(np.log(np.sum(np.exp(np.array(weighted_log_probs)), axis=0)))
        return log_likelihood
    
    def fit(self, X: np.ndarray, verbose: bool = True) -> 'GaussianMixtureModel':
        """
        Fit GMM to data using EM algorithm
        
        Args:
            X: Input data (n_samples, n_features)
            verbose: Print convergence messages
            
        Returns:
            self: Fitted model
        """
        self._initialize_parameters(X)
        self.log_likelihood_history = []
        
        logger.info(f"Starting EM algorithm for {self.max_iter} iterations")
        start_time = time.time()
        
        for i in range(self.max_iter):
            # E-step
            responsibilities = self._e_step(X)
            
            # M-step
            self._m_step(X, responsibilities)
            
            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(X)
            self.log_likelihood_history.append(log_likelihood)
            
            # Check convergence
            if i > 0:
                delta = np.abs(log_likelihood - self.log_likelihood_history[-2])
                if delta < self.tol:
                    self.converged = True
                    logger.info(f"Converged at iteration {i+1}, log-likelihood: {log_likelihood:.4f}")
                    break
            
            if verbose and (i % 10 == 0 or i == self.max_iter - 1):
                logger.info(f"Iteration {i+1}/{self.max_iter}, log-likelihood: {log_likelihood:.4f}")
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict posterior probabilities of components"""
        return self._e_step(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict component membership"""
        return np.argmax(self.predict_proba(X), axis=1)
    
    def sample(self, n_samples: int = 100) -> np.ndarray:
        """Generate samples from the fitted model"""
        if self.weights is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Select components based on weights
        component_indices = np.random.choice(
            self.n_components, 
            size=n_samples, 
            p=self.weights
        )
        
        samples = np.zeros((n_samples, self.means.shape[1]))
        for i in range(n_samples):
            comp_idx = component_indices[i]
            samples[i] = multivariate_normal.rvs(
                mean=self.means[comp_idx], 
                cov=self.covariances[comp_idx]
            )
        
        return samples, component_indices
    
    def bic(self, X: np.ndarray) -> float:
        """Calculate Bayesian Information Criterion"""
        n_samples, n_features = X.shape
        n_params = (
            self.n_components - 1 +  # weights
            self.n_components * n_features +  # means
            self.n_components * n_features * (n_features + 1) // 2  # covariances
        )
        
        log_likelihood = self.log_likelihood_history[-1]
        return -2 * log_likelihood + n_params * np.log(n_samples)
    
    def plot_convergence(self, save_path: Optional[str] = None):
        """Plot log-likelihood convergence"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.log_likelihood_history, 'o-', linewidth=2)
        plt.title('EM Algorithm Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Log-Likelihood')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved convergence plot to {save_path}")
        else:
            plt.show()
    
    def plot_results_1d(self, X: np.ndarray, save_path: Optional[str] = None):
        """Visualize results for 1D data (histogram + PDF)"""
        if X.shape[1] != 1:
            raise ValueError("This visualization only works for 1D data")
        
        plt.figure(figsize=(12, 8))
        
        # Plot histogram of data
        plt.hist(X, bins=50, density=True, alpha=0.5, color='skyblue', edgecolor='black')
        
        # Plot each component
        x_min, x_max = X.min() - 1, X.max() + 1
        x_vals = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
        
        total_pdf = np.zeros_like(x_vals)
        for k in range(self.n_components):
            pdf = self.weights[k] * norm.pdf(x_vals, self.means[k], np.sqrt(self.covariances[k]))
            plt.plot(x_vals, pdf, linewidth=2.5, label=f'Component {k+1}')
            total_pdf += pdf
        
        # Plot total mixture PDF
        plt.plot(x_vals, total_pdf, 'k--', linewidth=3, label='Mixture PDF')
        
        plt.title('Gaussian Mixture Model Fit')
        plt.xlabel('Value')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved 1D visualization to {save_path}")
        else:
            plt.show()
    
    def plot_results_2d(self, X: np.ndarray, save_path: Optional[str] = None):
        """Visualize results for 2D data (scatter plot + contours)"""
        if X.shape[1] != 2:
            raise ValueError("This visualization only works for 2D data")
        
        plt.figure(figsize=(12, 10))
        
        # Scatter plot of data points with cluster assignments
        y_pred = self.predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=30, alpha=0.6, edgecolor='k')
        
        # Plot component means
        plt.scatter(self.means[:, 0], self.means[:, 1], c='red', s=200, marker='X', edgecolor='black')
        
        # Plot covariance ellipses
        for k in range(self.n_components):
            cov = self.covariances[k]
            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.sqrt(lambda_)
            
            for j in range(1, 4):
                ellipse = plt.matplotlib.patches.Ellipse(
                    xy=(self.means[k, 0], self.means[k, 1]),
                    width=lambda_[0]*j*2, 
                    height=lambda_[1]*j*2,
                    angle=np.rad2deg(np.arccos(v[0, 0])),
                    fill=False,
                    linewidth=2,
                    linestyle='--',
                    alpha=0.7
                )
                plt.gca().add_patch(ellipse)
        
        plt.title('Gaussian Mixture Model Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved 2D visualization to {save_path}")
        else:
            plt.show()

# --------------------------
# Data Generation Functions
# --------------------------

def generate_gmm_data(
    n_samples: int = 1000,
    n_components: int = 3,
    n_features: int = 2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic GMM data"""
    np.random.seed(random_state)
    
    # Random parameters for GMM
    weights = np.random.dirichlet(np.ones(n_components))
    means = np.random.uniform(-10, 10, size=(n_components, n_features))
    
    # Generate random positive definite covariance matrices
    covariances = []
    for _ in range(n_components):
        A = np.random.randn(n_features, n_features)
        covariances.append(A.T @ A + np.eye(n_features))
    
    # Generate component assignments
    comp_assignments = np.random.choice(n_components, size=n_samples, p=weights)
    
    # Generate data
    X = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        comp_idx = comp_assignments[i]
        X[i] = multivariate_normal.rvs(mean=means[comp_idx], cov=covariances[comp_idx])
    
    return X, comp_assignments

def load_real_data() -> np.ndarray:
    """Load real dataset for demonstration"""
    from sklearn.datasets import load_iris
    data = load_iris()
    return data.data[:, :2]  # Use first two features for visualization

# --------------------------
# Main Workflow
# --------------------------

def main():
    """End-to-end demonstration of GMM with EM algorithm"""
    logger.info("Starting GMM Demonstration")
    
    # Configuration
    PLOT_DIR = "gmm_plots"
    N_COMPONENTS = 3
    N_FEATURES = 2
    N_SAMPLES = 1500
    
    # Create data
    synthetic_data, _ = generate_gmm_data(
        n_samples=N_SAMPLES,
        n_components=N_COMPONENTS,
        n_features=N_FEATURES
    )
    real_data = load_real_data()
    
    # Initialize and fit model
    gmm = GaussianMixtureModel(n_components=N_COMPONENTS, max_iter=50, tol=1e-4)
    
    logger.info("Fitting model to synthetic data")
    gmm.fit(synthetic_data)
    
    # Performance metrics
    logger.info(f"Final log-likelihood: {gmm.log_likelihood_history[-1]:.4f}")
    logger.info(f"BIC: {gmm.bic(synthetic_data):.4f}")
    
    # Visualization
    gmm.plot_convergence(f"{PLOT_DIR}/convergence.png")
    gmm.plot_results_2d(synthetic_data, f"{PLOT_DIR}/synthetic_clusters.png")
    
    # Generate new samples
    new_samples, comp_assignments = gmm.sample(n_samples=500)
    plt.figure(figsize=(10, 8))
    plt.scatter(new_samples[:, 0], new_samples[:, 1], c=comp_assignments, cmap='viridis')
    plt.title('Generated Samples from Fitted GMM')
    plt.savefig(f"{PLOT_DIR}/generated_samples.png", dpi=300)
    
    # Real data example
    logger.info("Fitting model to real data (Iris dataset)")
    gmm_real = GaussianMixtureModel(n_components=3, max_iter=50)
    gmm_real.fit(real_data)
    gmm_real.plot_results_2d(real_data, f"{PLOT_DIR}/iris_clusters.png")
    
    logger.info("Demonstration completed successfully")

if __name__ == "__main__":
    import os
    os.makedirs("gmm_plots", exist_ok=True)
    main()