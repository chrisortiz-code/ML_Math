import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def generate_gaussian_data(mean, cov, num_samples):
    """
    Generate data points from a multivariate Gaussian distribution.

    Parameters:
        mean (list): Mean vector of the Gaussian distribution.
        cov (np.ndarray): Covariance matrix of the Gaussian distribution.
        num_samples (int): Number of samples to generate.

    Returns:
        np.ndarray: Generated data points.
    """
    return np.random.multivariate_normal(mean, cov, num_samples)

def plot_gaussian_heatmap(data, resolution=100, kernel_bandwidth=0.2):
    """
    Plot a heatmap showing the density of Gaussian-distributed data points.

    Parameters:
        data (np.ndarray): Data points (num_samples x 2).
        resolution (int): Resolution of the heatmap grid.
        kernel_bandwidth (float): Bandwidth for Gaussian kernel smoothing.
    """
    x_min, y_min = np.min(data, axis=0) - 1
    x_max, y_max = np.max(data, axis=0) + 1

    # Create a grid for plotting
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    x_grid, y_grid = np.meshgrid(x, y)

    # Compute density using a Gaussian kernel
    pos = np.dstack((x_grid, y_grid))
    kde = multivariate_normal(mean=np.mean(data, axis=0), cov=kernel_bandwidth**2)
    density = kde.pdf(pos)

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.contourf(x_grid, y_grid, density, levels=50, cmap="magma")
    plt.colorbar(label="Density")
    plt.scatter(data[:, 0], data[:, 1], c="white", s=10, alpha=0.7, label="Data Points")
    plt.title("Gaussian Heatmap with Data Points")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.show()

# Example usage
mean = [0, 0]  # Mean vector (center of Gaussian)
cov = [[1, 0.5], [0.5, 1]]  # Covariance matrix (controls spread and orientation)
data = generate_gaussian_data(mean, cov, num_samples=500)  # Generate Gaussian data

# Plot the heatmap
plot_gaussian_heatmap(data, resolution=100, kernel_bandwidth=0.5)
