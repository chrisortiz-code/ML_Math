import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_and_analyze_distribution(dimensions, num_samples=1000, mean_range=(-5, 5), scale=1):
    """
    Generates a random sample, computes its covariance matrix, and visualizes it.

    Parameters:
        dimensions (int): Number of dimensions for the random data.
        num_samples (int): Number of samples to generate.
        mean_range (tuple): Range for generating random means.
        scale (float): Scale factor for random data generation.

    Returns:
        tuple: Generated samples (np.ndarray), computed covariance matrix (np.ndarray).
    """
    num_samples = max(1, num_samples)  # Ensure at least one sample is generated

    # Generate random mean vector and data
    mean = np.random.uniform(mean_range[0], mean_range[1], dimensions)
    data = np.random.randn(num_samples, dimensions) * scale + mean  # Add mean to center data

    # Check generated data
    print("Generated data shape:", data.shape)

    # Compute covariance matrix (only if num_samples > 1)
    cov_matrix = np.cov(data, rowvar=False) if num_samples > 1 else np.array([[0]])

    # Visualization for 2D and 3D distributions
    plt.figure(figsize=(8, 6))
    if dimensions == 2:
        plt.scatter(data[:, 0], data[:, 1], alpha=0.7, s=10)
        plt.title("Generated 2D Multivariate Distribution")
        plt.xlabel("X1")
        plt.ylabel("X2")
    elif dimensions == 3:
        ax = plt.subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.7, s=10)
        ax.set_title("Generated 3D Multivariate Distribution")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("X3")
    else:
        print(f"Visualization for {dimensions} dimensions is not supported.")

    plt.grid(alpha=0.3)
    plt.show()

    print(f"Computed Covariance Matrix:\n{cov_matrix}")
    return data, cov_matrix


# Example usage
data, cov_matrix = generate_and_analyze_distribution(dimensions=2, num_samples=1, mean_range=(-3, 3), scale=2)

# Example usage
data, cov_matrix = generate_and_analyze_distribution(dimensions=2, num_samples=1000, mean_range=(-3, 3), scale=2)

