"""
encoding.py
===========
QML Series 2 | Post 06 -> Data Encoding Layer

Converts MNIST images into quantum rotation angles.
Two strategies are provided:
  1. strided_encode  -> crude strided sampling (used in Posts 6-7)
  2. variance_encode -> variance-based feature selection (used in Post 9)
"""

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Strategy 1: Strided Sampling (Post 06)
# ---------------------------------------------------------------------------

def encode_data(image: torch.Tensor, n_qubits: int = 16) -> torch.Tensor:
    """
    Encode a single MNIST image into rotation angles using strided sampling.

    Parameters
    ----------
    image     : torch.Tensor  shape (1, 28, 28)  or  (28, 28)
    n_qubits  : int  -> number of qubits / features to extract

    Returns
    -------
    torch.Tensor of shape (n_qubits,)  with values in [0, pi]

    How it works
    ------------
    Step 1 -> Flatten the 28x28 image into a 784-element vector.
    Step 2 -> Take every (784 // n_qubits)-th pixel -> n_qubits values.
              This is "strided sampling": we space out our samples evenly.
    Step 3 -> Pixel values are in [0, 1] after torchvision normalization.
              Multiply by pi -> rotation angles in [0, pi].
    """
    # Step 1: flatten to 1-D
    flat = image.view(-1)                        # shape: (784,)

    # Step 2: stride-sample down to n_qubits features
    stride = len(flat) // n_qubits               # e.g. 784 // 16 = 49
    features = flat[::stride][:n_qubits]         # shape: (n_qubits,)

    # Step 3: map pixel values [0,1] -> rotation angles [0, pi]
    angles = features * torch.tensor(np.pi)      # shape: (n_qubits,)

    return angles.float()


# ---------------------------------------------------------------------------
# Strategy 2: Variance-Based Feature Selection (Post 09)
# ---------------------------------------------------------------------------

def compute_top_variance_indices(dataloader, n_qubits: int = 16,
                                  n_samples: int = 500) -> torch.Tensor:
    """
    Find the n_qubits pixel positions with the highest variance
    across a sample of training images.

    High-variance pixels change the most between digit classes and
    therefore carry the most discriminative information.

    Parameters
    ----------
    dataloader : DataLoader   -> training data loader
    n_qubits   : int          -> how many top pixels to select
    n_samples  : int          -> how many images to use for variance estimate

    Returns
    -------
    torch.Tensor of shape (n_qubits,)  containing pixel indices
    """
    all_pixels = []

    for images, _ in dataloader:
        for img in images:
            all_pixels.append(img.view(-1))      # flatten each image
            if len(all_pixels) >= n_samples:
                break
        if len(all_pixels) >= n_samples:
            break

    # Stack into matrix: shape (n_samples, 784)
    pixel_matrix = torch.stack(all_pixels)

    # Compute per-pixel variance across all samples
    variances = pixel_matrix.var(dim=0)          # shape: (784,)

    # Return indices of the n_qubits highest-variance pixels
    top_indices = torch.argsort(variances, descending=True)[:n_qubits]
    return top_indices


def encode_data_variance(image: torch.Tensor,
                          top_indices: torch.Tensor) -> torch.Tensor:
    """
    Encode an MNIST image using pre-computed variance-based pixel indices.

    Parameters
    ----------
    image       : torch.Tensor  shape (1, 28, 28)
    top_indices : torch.Tensor  shape (n_qubits,)  -> from compute_top_variance_indices

    Returns
    -------
    torch.Tensor of shape (n_qubits,) with values in [0, pi]

    How it works
    ------------
    Instead of every 49th pixel, we pick the n_qubits pixels that have
    the most variance across the training set, the most "informative" pixels.
    Then we do the same [0,1] -> [0,pi] mapping as strided encoding.
    """
    flat = image.view(-1)                        # shape: (784,)
    features = flat[top_indices]                 # shape: (n_qubits,)
    angles = features * torch.tensor(np.pi)      # map to [0, pi]
    return angles.float()