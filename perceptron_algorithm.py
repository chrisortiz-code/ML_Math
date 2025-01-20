import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import imageio.v2 as imageio

def get_random_data(num_points=100, d_sep=1.5, x_range=(0, 10), y_range=(-10, 20)):
    """
    Generate random points outside a separation barrier around a random line.

    Parameters:
        num_points (int): Number of points to generate.
        d_sep (float): Distance of the barrier from the line.
        x_range (tuple): Range of x-coordinates for random points.
        y_range (tuple): Range of y-coordinates for random points.

    Returns:
        tuple: A tuple containing:
            - points (ndarray): Array of shape (num_points, 2) with generated points.
            - labels (ndarray): Array of shape (num_points,) with labels (1 or -1).
            - m (float): Slope of the separating line.
            - h (float): Intercept of the separating line.
    """
    # Random line parameters
    m = np.random.uniform(-2, 2)  # Random slope
    h = np.random.uniform(-5, 5)  # Random intercept

    # Generate random x and y coordinates
    x_points = np.random.uniform(x_range[0], x_range[1], num_points * 10)
    y_points = np.random.uniform(y_range[0], y_range[1], num_points * 10)

    # Define barrier bounds
    denom = np.sqrt(m**2 + 1)
    upper_bound = lambda x: m * x + h + d_sep * denom
    lower_bound = lambda x: m * x + h - d_sep * denom

    # Filter points outside the barrier
    points = []
    labels = []
    for x, y in zip(x_points, y_points):
        if y > upper_bound(x):  # Above the upper barrier
            points.append((x, y))
            labels.append(1)  # Label for points above
        elif y < lower_bound(x):  # Below the lower barrier
            points.append((x, y))
            labels.append(-1)  # Label for points below
        if len(points) >= num_points:  # Stop when enough points are collected
            break

    return np.array(points), np.array(labels), m, h


class Perceptron:
    """
    Perceptron algorithm for binary classification.
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.w = None
        self.b = 0

    def initialize_weights(self, input_dim):
        """
        Initialize weights and bias.
        """
        self.w = np.random.uniform(-1, 1, input_dim)
        self.b = np.random.uniform(-1, 1)

    def update(self, x, y):
        """
        Update weights and bias for a single data point.

        Parameters:
            x (ndarray): Feature vector of the data point.
            y (int): Label of the data point (+1 or -1).
        """
        prediction = np.sign(np.dot(self.w, x) + self.b)
        if prediction != y:
            self.w += self.learning_rate * y * x
            self.b += self.learning_rate * y

    def predict(self, x):
        """
        Predict the label of a data point.

        Parameters:
            x (ndarray): Feature vector of the data point.

        Returns:
            int: Predicted label (+1 or -1).
        """
        return np.sign(np.dot(self.w, x) + self.b)


def plot_perceptron_decision_boundary(points, labels, perceptron, iteration, folder, x_range=(0, 10)):
    """
    Plot the data and the decision boundary learned by the Perceptron and save as an image.

    Parameters:
        points (ndarray): Array of shape (num_points, 2) with generated points.
        labels (ndarray): Array of shape (num_points,) with labels (1 or -1).
        perceptron (Perceptron): Trained Perceptron object.
        iteration (int): Current iteration number.
        folder (str): Folder to save the plots.
        x_range (tuple): Range of x-coordinates for the plot.
    """
    # Select only the data points seen by the perceptron up to this iteration
    points_seen = points[:iteration]
    labels_seen = labels[:iteration]

    # Separate points by label
    x_class1 = points_seen[labels_seen == 1][:, 0]
    y_class1 = points_seen[labels_seen == 1][:, 1]
    x_class2 = points_seen[labels_seen == -1][:, 0]
    y_class2 = points_seen[labels_seen == -1][:, 1]

    # Decision boundary
    x_line = np.linspace(x_range[0], x_range[1], 100)
    if perceptron.w[1] != 0:  # Avoid division by zero
        y_line = -(perceptron.w[0] * x_line + perceptron.b) / perceptron.w[1]
    else:
        y_line = np.full_like(x_line, -perceptron.b / perceptron.w[0])

    # Plot points
    plt.scatter(x_class1, y_class1, color='blue', label='Class 1 (Label = 1)')
    plt.scatter(x_class2, y_class2, color='red', label='Class 2 (Label = -1)')
    plt.plot(x_line, y_line, color='green', linestyle='--', label='Learned Decision Boundary')

    # Add labels and legend
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(f'Perceptron Decision Boundary (Iteration {iteration})')
    plt.legend()
    plt.grid()

    # Save plot
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, f'iteration_{iteration:03d}.png'))
    plt.close()

def create_gif_from_folder(folder, output_gif, duration=0.5):
    """
    Create a GIF from images in a folder.

    Parameters:
        folder (str): Path to the folder containing images.
        output_gif (str): Path for the output GIF file.
        duration (float): Duration for each frame in the GIF (in seconds).
    """
    # Get all image files in the folder, sorted by name
    images = sorted(
        [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".png")]
    )

    # Read and combine images into a GIF
    with imageio.get_writer(output_gif, mode="I", duration=duration) as writer:
        for image_path in images:
            image = imageio.imread(image_path)
            writer.append_data(image)

    print(f"GIF saved to {output_gif}")

# Example usage
if __name__ == "__main__":
    # Folder to save plots
    output_folder = "perceptron_plots"

    # Clear the folder if it exists
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    # Generate random data
    points, labels, m, h = get_random_data(num_points=100, d_sep=1.5)

    # Initialize the Perceptron
    perceptron = Perceptron(learning_rate=0.1)
    perceptron.initialize_weights(points.shape[1])

    # Train the Perceptron and save plots
    for i, (x, y) in enumerate(zip(points, labels), start=1):
        perceptron.update(x, y)
        plot_perceptron_decision_boundary(points, labels, perceptron, i, output_folder)

    # Create GIF
    output_gif_path = "perceptron_training.gif"
    create_gif_from_folder(output_folder, output_gif_path, duration=0.2)

    print(f"Final weights: {perceptron.w}")
    print(f"Final bias: {perceptron.b}")
    print(f"Plots saved to folder: {output_folder}")
