# Gaussian Mixture Model with EM Algorithm

## Overview
This repository implements a Gaussian Mixture Model (GMM) using the Expectation-Maximization (EM) algorithm for maximum likelihood estimation. The code includes:
1. Complete EM algorithm implementation
2. Synthetic data generation
3. Model evaluation (BIC, log-likelihood)
4. Comprehensive visualization tools
5. Real dataset integration (Iris dataset)

## Key Features
- **Parameter Estimation**: Automatically estimates weights, means, and covariances
- **Convergence Tracking**: Monitors log-likelihood during optimization
- **Visualization**:
  - EM convergence curve
  - 1D probability density visualization
  - 2D cluster visualization with covariance ellipses
- **Model Selection**: Bayesian Information Criterion (BIC) calculation
- **Data Generation**: Synthetic GMM data generation
- **Sampling**: Generate new samples from fitted model

## Mathematical Foundations
The Gaussian Mixture Model probability density function:
$$p(x|\theta) = \sum_{k=1}^K \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)$$

**EM Algorithm**:
1. **E-step**: Compute responsibilities
   $$\gamma(z_{nk}) = \frac{\pi_k \mathcal{N}(x_n|\mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_n|\mu_j, \Sigma_j)}$$
   
2. **M-step**: Update parameters
   $$\mu_k^{\text{new}} = \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) x_n$$
   $$\Sigma_k^{\text{new}} = \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) (x_n - \mu_k^{\text{new}})(x_n - \mu_k^{\text{new}})^\top$$
   $$\pi_k^{\text{new}} = \frac{N_k}{N}$$
   where $N_k = \sum_{n=1}^N \gamma(z_{nk})$

## Usage
**Initialize Model**:
   ```python
   gmm = GaussianMixtureModel(n_components=3, max_iter=100)
   ```
## Fit to Data
```python
gmm.fit(X)
```
## Make Predictions:**
```python
probabilities = gmm.predict_proba(X)
labels = gmm.predict(X)
```
##Visualize Results:
```python
gmm.plot_convergence()
gmm.plot_results_2d(X)
```
##Generate Samples:
```python
    samples, _ = gmm.sample(n_samples=1000)
```
##Requirements

    Python 3.7+

    NumPy

    SciPy

    Matplotlib

    scikit-learn (for real dataset loading)

    Seaborn (for visualization)

##Output Examples

    Convergence Plot:
    https://gmm_plots/convergence.png

    2D Clustering:
    https://gmm_plots/synthetic_clusters.png

    Generated Samples:
    https://gmm_plots/generated_samples.png


