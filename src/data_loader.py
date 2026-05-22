"""
data_loader.py
==============
QML Series 2 | Post 06 -> MNIST Data Loading

Downloads and prepares the MNIST dataset using torchvision.
Returns DataLoader objects ready for training and evaluation.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_mnist_loaders(train_size: int = 500,
                      test_size:  int = 100,
                      batch_size: int = 10,
                      data_dir:   str = "./data") -> tuple:
    """
    Download MNIST and return train + test DataLoaders.

    Parameters
    ----------
    train_size : int  -> number of training images to use
                        (500 for Posts 6-8, 1000 for Post 9 optimization)
    test_size  : int  -> number of test images to evaluate on
    batch_size : int  -> images per batch during training
    data_dir   : str  -> where to store the downloaded data

    Returns
    -------
    (train_loader, test_loader) -> PyTorch DataLoader objects

    Why these transforms?
    ---------------------
    transforms.ToTensor()
        Converts PIL image (H, W) with pixel values [0, 255]
        to a torch.Tensor of shape (1, 28, 28) with values [0, 1].
        This is the format our encode_data() function expects.

    transforms.Normalize((0.1307,), (0.3081,))
        Normalizes pixel values using the MNIST dataset mean (0.1307)
        and standard deviation (0.3081).
        After normalization most values sit in roughly [-1, +1],
        which makes the subsequent encoding into [0, pi] approximate
        angles distributed around pi/2, a balanced starting point.

    Why only 500 training images?
    ------------------------------
    Training a quantum circuit is ~80x slower than a classical network
    of comparable size (measured in Post 08). Using a small subset keeps
    experiments tractable on a laptop CPU simulator while still showing
    meaningful learning. Post 09 doubles this to 1000 images.
    """

    # Define the preprocessing pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download the full MNIST training set (60,000 images) to data_dir
    full_train = datasets.MNIST(
        root=data_dir, train=True,
        download=True, transform=transform
    )

    # Download the full MNIST test set (10,000 images) to data_dir
    full_test = datasets.MNIST(
        root=data_dir, train=False,
        download=True, transform=transform
    )

    # Take only the first train_size and test_size images
    # This is a deliberate constraint for simulation speed
    train_subset = Subset(full_train, range(train_size))
    test_subset  = Subset(full_test,  range(test_size))

    # Wrap in DataLoaders for batched iteration
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True    # shuffle each epoch to avoid ordering bias
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False   # keep test set in fixed order for reproducibility
    )

    print(f"  MNIST loaded -> Train: {train_size} images | "
          f"Test: {test_size} images | Batch size: {batch_size}")

    return train_loader, test_loader