import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import shutil
import imageio.v2 as imageio


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # Define weights and bias as parameters
        self.weights = nn.Parameter(torch.zeros(1))  # Initialize weights to 0
        self.bias = nn.Parameter(torch.zeros(1))     # Initialize bias to 0

    def forward(self, x):
        # Linear regression formula
        return self.weights * x + self.bias


def save_plot_with_predictions(X, y, train_X, train_y, test_X, predictions, model, epoch, loss, folder):
    """
    Save a plot of the data, predictions, and learned line at a specific epoch.

    Parameters:
        X (torch.Tensor): All input data (for full line plotting).
        y (torch.Tensor): All true labels (for full line plotting).
        train_X (torch.Tensor): Training input data.
        train_y (torch.Tensor): Training labels.
        test_X (torch.Tensor): Test input data (for predictions).
        predictions (torch.Tensor): Predicted values for test_X.
        model (LinearRegressionModel): The linear regression model.
        epoch (int): Current epoch number.
        loss (float): Current loss value.
        folder (str): Folder to save the plot.
    """
    plt.figure(figsize=(8, 6))

    # Plot true data
    plt.scatter(train_X.numpy(), train_y.numpy(), color="blue", label="Training Data")
    plt.scatter(test_X.numpy(), predictions.numpy(), color="red", label="Predictions")

    # Plot learned line
    full_X = torch.linspace(0, X.max(), 100).reshape(-1, 1)
    learned_line = model(full_X)
    plt.plot(full_X.numpy(), learned_line.detach().numpy(), color="green", label="Learned Line")

    # Add epoch and loss to the plot
    plt.text(0.1, 0.9, f"Epoch: {epoch}", transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.1, 0.85, f"Loss: {loss:.4f}", transform=plt.gca().transAxes, fontsize=12)

    # Add labels and legend
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression: Predictions and Learned Line")
    plt.legend()
    plt.grid()

    # Save the plot
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, f"epoch_{epoch:03d}.png"))
    plt.close()


def create_gif_from_folder(folder, output_gif, duration=0.5):
    """
    Create a GIF from images in a folder.

    Parameters:
        folder (str): Path to the folder containing images.
        output_gif (str): Path for the output GIF file.
        duration (float): Duration for each frame in the GIF (in seconds).
    """
    images = sorted(
        [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".png")]
    )
    with imageio.get_writer(output_gif, mode="I", duration=duration) as writer:
        for image_path in images:
            image = imageio.imread(image_path)
            writer.append_data(image)
    print(f"GIF saved to {output_gif}")


# Example usage
if __name__ == "__main__":
    output_folder = "linear_regression_plots"
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    # Generate synthetic data
    torch.manual_seed(42)
    n_samples = 15
    X = torch.arange(0, n_samples, dtype=torch.float32).reshape(-1, 1)
    true_slope = 2.5
    true_intercept = 1.0
    y = true_slope * X + true_intercept

    # Split data
    train_X = X[:10]
    train_y = y[:10]
    test_X = X[10:]

    # Initialize model and optimizer
    model = LinearRegressionModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop with early stopping
    epochs = 500
    threshold = 1e-6  # Minimum change in weights/bias to continue training
    prev_weights = model.weights.clone().detach()
    prev_bias = model.bias.clone().detach()

    for epoch in range(1, epochs + 1):
        # Forward pass
        y_pred = model(train_X)
        loss = ((y_pred - train_y) ** 2).mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Predict next points
        with torch.no_grad():
            predictions = model(test_X)

        # Save plot
        save_plot_with_predictions(X, y, train_X, train_y, test_X, predictions, model, epoch, loss.item(), output_folder)

        # Check for early stopping
        weight_change = (model.weights - prev_weights).abs().max()
        bias_change = (model.bias - prev_bias).abs().max()

        if max(weight_change, bias_change) < threshold:
            print(f"Stopping early at epoch {epoch} due to minimal changes in parameters.")
            break

        prev_weights = model.weights.clone().detach()
        prev_bias = model.bias.clone().detach()

    # Create GIF
    output_gif_path = "linear_regression_training.gif"
    create_gif_from_folder(output_folder, output_gif_path, duration=0.2)

    print(f"Plots saved to folder: {output_folder}")
    print(f"GIF saved to: {output_gif_path}")
