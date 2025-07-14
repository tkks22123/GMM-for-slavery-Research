This repository implements a Gaussian Mixture Model (GMM) using the Expectation-Maximization (EM) algorithm for maximum likelihood estimation for modern slavery research. The code includes:

    Complete EM algorithm implementation

    Synthetic data generation

    Model evaluation (BIC, log-likelihood)

    Comprehensive visualization tools

    Real dataset integration (Iris dataset)

Key Features

    Parameter Estimation: Automatically estimates weights, means, and covariances

    Convergence Tracking: Monitors log-likelihood during optimization

    Visualization:

        EM convergence curve

        1D probability density visualization

        2D cluster visualization with covariance ellipses

    Model Selection: Bayesian Information Criterion (BIC) calculation

    Data Generation: Synthetic GMM data generation

    Sampling: Generate new samples from fitted model

Mathematical Foundations

The Gaussian Mixture Model probability density function:
p(x∣θ)=∑k=1KπkN(x∣μk,Σk)p(x∣θ)=∑k=1K​πk​N(x∣μk​,Σk​)

EM Algorithm:

    E-step: Compute responsibilities
    γ(znk)=πkN(xn∣μk,Σk)∑j=1KπjN(xn∣μj,Σj)γ(znk​)=∑j=1K​πj​N(xn​∣μj​,Σj​)πk​N(xn​∣μk​,Σk​)​

    M-step: Update parameters
    μknew=1Nk∑n=1Nγ(znk)xnμknew​=Nk​1​∑n=1N​γ(znk​)xn​
    Σknew=1Nk∑n=1Nγ(znk)(xn−μknew)(xn−μknew)⊤Σknew​=Nk​1​∑n=1N​γ(znk​)(xn​−μknew​)(xn​−μknew​)⊤
    πknew=NkNπknew​=NNk​​
    where $N_k = \sum_{n=1}^N \gamma(z_{nk})$

Usage

    Initialize Model:
    python

gmm = GaussianMixtureModel(n_components=3, max_iter=100)

Fit to Data:
python

gmm.fit(X)

Make Predictions:
python

probabilities = gmm.predict_proba(X)
labels = gmm.predict(X)

Visualize Results:
python

gmm.plot_convergence()
gmm.plot_results_2d(X)

Generate Samples:
python

    samples, _ = gmm.sample(n_samples=1000)

Requirements

    Python 3.7+

    NumPy

    SciPy

    Matplotlib

    scikit-learn (for real dataset loading)

    Seaborn (for visualization)

Output Examples

    Convergence Plot:
    https://gmm_plots/convergence.png

    2D Clustering:
    https://gmm_plots/synthetic_clusters.png

    Generated Samples:
    https://gmm_plots/generated_samples.png

Performance Notes

    Algorithm complexity: O(k·n·d²) per iteration

    Convergence typically in 20-50 iterations

    Regularization ensures numerical stability

    Automatic covariance regularization prevents singularities

Extensions

    Add different covariance types (spherical, diagonal)

    Implement variational Bayesian Gaussian mixture

    Add parallel computation for E-step

    Incorporate feature importance analysis

This implementation provides a robust foundation for probabilistic modeling with mixture distributions and demonstrates core principles of maximum likelihood estimation through the EM algorithm.
