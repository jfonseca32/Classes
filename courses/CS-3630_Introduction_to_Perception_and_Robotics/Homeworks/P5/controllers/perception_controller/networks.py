import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from IPython.display import HTML, display
from torch.autograd import Variable

""" Stores classes for networks and training functions. """


class LeNet(nn.Module):
    def __init__(self) -> None:
        """Initialize the LeNet layers for grayscale digit classification.

        Notes
        -----
        Network accepts images of shape (N, 1, 32, 32) and applies
        three convolutional layers, two average-pooling operations,
        and two fully connected layers.

        ReLU follows each convolutional layer and the first fully
        connected layer. The final layer produces 10 raw class scores (0 through 9).
        """
        super().__init__()

        # 6 filters of shape (1, 5, 5); output: (N, 6, 28, 28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)

        # 16 filters of shape (6, 5, 5); after first pool, output: (N, 16, 10, 10)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)

        # 120 filters of shape (16, 5, 5); after second pool, output: (N, 120, 1, 1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)

        # combines 120 input features into 84 learned features; output: (N, 84)
        self.linear1 = nn.Linear(in_features=120, out_features=84)

        # Produces one raw score for each digit class; output: (N, 10)
        self.linear2 = nn.Linear(in_features=84, out_features=10)

        # averages each 2×2 window independently within each channel.
        # first use: (N, 6, 28, 28) → (N, 6, 14, 14)
        # second use: (N, 16, 10, 10) → (N, 16, 5, 5)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # max(number, 0); preserves the tensor's shape
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass with the network.

        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (N,C,H,W).

        Returns
        -------
        torch.Tensor
            The output (raw scores) of shape (N,10).
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.reshape(x.shape[0], -1)  # (N, 120, 1, 1) -> (N, 120)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def plot_losses(train_losses: list[float], test_losses: list[float]) -> None:
    """Display training and test loss curves by epoch.

    Parameters
    ----------
    train_losses : list of float
        Training loss for each epoch, in chronological order.
    test_losses : list of float
        Test loss for each epoch, in chronological order.

    Returns
    -------
    None
        Displays the plot without returning a value.

    Notes
    -----
    Epoch indices start at zero. The plot includes a title, labeled
    axes, a grid, and a legend identifying each loss curve.
    """
    fig, ax = plt.subplots()

    ax.plot(train_losses, label="training loss", color="blue", linewidth=2, linestyle="-")
    ax.plot(test_losses, label="testing loss", color="red", linewidth=2, linestyle="--")
    ax.set_title("Train and Test Losses vs. Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error Loss")
    ax.grid(True)
    ax.legend()

    plt.show()


class ImageClassifier:
    """
    A class that wraps a saved LeNet model and provides a method to classify an image.
    """

    def __init__(self, model_path=None) -> None:
        """
        Initialize the ImageClassifier. A wrapper around a trained LeNet model that provides
        infrastructure for classification.
        """
        self.model = LeNet()
        if model_path:
            try:
                self.model.load_state_dict(load_model(model_path))
            except Exception as e:
                print(f"Error loading model: {e}")
                raise FileNotFoundError(f"Model not found at {model_path}") from e
            self.model.eval()  # turn off gradients
        else:
            print("[WARNING] no model path specified, inference will not work.")

        # Find Device to use
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")

        self.model.to(self.device)

        # MNIST: class index i is the handwritten digit i
        self.labels_dict = {i: str(i) for i in range(10)}

    def preprocess_image(self, numpy_image: np.ndarray) -> torch.Tensor:
        """Prepare an RGB camera image for LeNet inference.

        Parameters
        ----------
        numpy_image : numpy.ndarray
            RGB image of shape (H, W, 3), with pixel values in [0, 255].

        Returns
        -------
        torch.Tensor
            Grayscale image of shape (1, 1, 32, 32), with dtype
            ``torch.float32`` and pixel values scaled to [0, 1].
            The first two dimensions represent batch size and channels.

        Notes
        -----
        Converts RGB to grayscale, resizes to 32 by 32 pixels, and
        scales pixel values to match the training data. The returned
        tensor is on the CPU; device placement is handled during inference.
        """
        numpy_image = numpy_image.astype(np.uint8)

        # Convert RGB (H, W, 3) to Grayscale (H, W)
        gray = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)

        # Resize to model's expected dimensions (32, 32)
        gray = cv2.resize(gray, (32, 32))

        # Convert pixel values from [0, 255] to float32 in [0, 1]
        image_tensor = torch.from_numpy(gray).to(torch.float32) / 255.0

        # (32, 32) -> (1, 32, 32) -> (1, 1, 32, 32)
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

        return image_tensor

    def single_inference(self, image: np.ndarray) -> int:
        """Predict the digit class for a single RGB camera image.

        Parameters
        ----------
        image : numpy.ndarray
            RGB image of shape (H, W, 3), with pixel values in [0, 255].

        Returns
        -------
        int
            Predicted digit class index, from 0 to 9.

        Notes
        -----
        Preprocesses the image and moves it to the model's device.
        Runs the model in evaluation mode with gradient tracking disabled,
        then selects the class with the highest raw score.
        """
        # Convert to (1, 1, 32, 32) tensor and move to model's device
        image_tensor = self.preprocess_image(image).to(self.device)

        self.model.eval()
        with (
            torch.no_grad()
        ):  # Disable gradient calcs since don't need it as inference doesn't update weights
            logits = self.model(image_tensor)  # shape: (1, 10)
            predicted = logits.argmax(dim=1).item()  # select highest scoring class

        return predicted

    def class_to_character(self, class_index: int) -> str:
        """Convert a predicted digit class index to its character.

        Parameters
        ----------
        class_index : int
            Predicted digit class index, from 0 to 9.

        Returns
        -------
        str
            String representation of the digit, such as ``"3"``.
        """
        return str(class_index)

    def inference(self, image):
        """
        Classify a single image from Webots robot camera.
        """
        predicted_class_index = self.single_inference(image)
        character = self.class_to_character(predicted_class_index)
        print(f"Predicted character: {character}, class index: {predicted_class_index}")
        return character


##### TRAINING HELPER FUNCTIONS DEFINED BY STAFF #####


class ProgressMonitor:
    tmpl = """
        <p>Loss: {loss:0.4f}   {value} / {length}</p>
        <progress value='{value}' max='{length}', style='width: 100%'>{value}</progress>
    """

    def __init__(self, length):
        self.length = length
        self.count = 0
        self.display = display(self.html(0, 0), display_id=True)

    def html(self, count, loss):
        return HTML(self.tmpl.format(length=self.length, value=count, loss=loss))

    def update(self, count, loss):
        self.count += count
        self.display.update(self.html(self.count, loss))


def train_iter(model, train_dataloader, optimizer, criterion, dataset_sizes, device):
    """One iteration of training through the entire dataset."""
    running_loss = 0.0

    # set model to training mode
    model.train()

    # create a progress bar
    progress = ProgressMonitor(length=dataset_sizes["train"])

    for data in train_dataloader:
        inputs, labels = data
        batch_size = inputs.shape[0]

        inputs = Variable(inputs.to(device))
        labels = Variable(labels.to(device))

        # zero out previous gradients
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.data * batch_size
        # update progress bar
        progress.update(batch_size, running_loss)

    return running_loss


def test_iter(model, test_dataloader, optimizer, criterion, device):
    """Test the current model on the test dataset."""
    # set model to evaluation mode
    model.eval()
    # Do not need gradients for testing
    test_loss = 0.0
    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = data
            batch_size = inputs.shape[0]

            inputs = Variable(inputs.to(device))
            labels = Variable(labels.to(device))
            outputs = model(inputs)

            # calculate the loss
            optimizer.zero_grad()
            loss = criterion(outputs, labels)

            # update running loss value
            test_loss += loss.data * batch_size
    return test_loss


def train_and_test(
    model, criterion, optimizer, num_epochs, dataloaders, dataset_sizes, device, first_epoch=1
):
    """Train and test the neural network on the specified dataset for several iterations."""
    plot_train_loss = []
    plot_test_loss = []

    model.to(device)

    for epoch in range(first_epoch, first_epoch + num_epochs):
        print("\nEpoch", epoch)

        # training
        running_loss = train_iter(
            model, dataloaders[0], optimizer, criterion, dataset_sizes, device
        )

        epoch_loss = running_loss / dataset_sizes["train"]
        print("Training loss:", epoch_loss.item())
        plot_train_loss.append(epoch_loss.cpu())

        # testing
        test_loss = test_iter(model, dataloaders[-1], optimizer, criterion, device)

        epoch_test_loss = test_loss / dataset_sizes["test"]
        print("Testing loss:", epoch_test_loss.item())
        plot_test_loss.append(epoch_test_loss.cpu())

    return plot_train_loss, plot_test_loss, model


def accuracy(loader, model, train=True, device=None):
    """
    Calculates prediction accuracy of the model on data

    Args:
        loader: (Dataloader) contains data and labels
        model: neural network model
        train (boolean): True if training, False if testing
    """
    if device is None:
        device = torch.device("cpu")
    num_correct = num_samples = 0
    model.eval()
    with torch.no_grad():
        for data in loader:
            inputs, labels = data

            inputs = Variable(inputs.to(device))
            labels = Variable(labels.to(device))
            outputs = model(inputs)
            # prediction with the top score
            _, preds = outputs.max(1)
            num_correct += (preds == labels).sum()
            num_samples += preds.size(0)
    accuracy = (num_correct.item() / num_samples) * 100

    if train:
        print(
            f"Model Predicted {num_correct.item()} correctly out of {num_samples} "
            f"from training dataset, Accuracy : {accuracy:.2f}"
        )
    else:
        print(
            f"Model Predicted {num_correct.item()} correctly out of {num_samples} "
            f"from testing dataset, Accuracy : {accuracy:.2f}"
        )


def load_model(model_path):
    """
    Load the model from the disk if it exists. Returns the model state dictionary.
    """
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        return checkpoint["model_state_dict"]
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")
