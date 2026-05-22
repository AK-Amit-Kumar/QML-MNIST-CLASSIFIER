"""
model.py
========
QML Series 2 | Post 07 -> Hybrid Quantum-Classical Model

Combines the variational quantum circuit (circuit.py) with a classical
linear output layer to form a complete PyTorch model for MNIST classification.

Architecture:
  Input image (28x28)
      |
  encode_data()              <- classical pre-processing (encoding.py)
      |
  quantum_circuit(inputs, weights)   <- 16-qubit variational circuit
      |
  16 expectation values in [-1, +1]
      |
  nn.Linear(16 -> 10)        <- classical output layer
      |
  10 class logits  (digit 0-9)
"""

import torch
import torch.nn as nn
from src.circuit import quantum_circuit, get_weight_shape, N_QUBITS
from src.encoding import encode_data


# ---------------------------------------------------------------------------
# Hybrid Model
# ---------------------------------------------------------------------------

class HybridQMLModel(nn.Module):
    """
    A hybrid quantum-classical neural network for MNIST digit classification.

    Parameters
    ----------
    n_qubits  : int  -> number of qubits in the quantum circuit (default 16)
    n_layers  : int  -> number of variational layers (default 2)

    Attributes
    ----------
    weights         : nn.Parameter  -> quantum circuit weights, shape (n_layers, n_qubits)
    classical_layer : nn.Linear     -> maps 16 quantum outputs -> 10 class logits

    Why nn.Parameter?
    -----------------
    Wrapping the weights tensor in nn.Parameter tells PyTorch to:
      * Include them in model.parameters()  (so the optimizer sees them)
      * Track gradients through them automatically
    PennyLane then uses the parameter-shift rule to compute those gradients
    through the quantum circuit -> PyTorch and PennyLane work together here.
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = 2,
                 use_variance_encoding: bool = False,
                 top_indices: torch.Tensor = None):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_variance_encoding = use_variance_encoding
        self.top_indices = top_indices   # used only when variance encoding is on

        # Quantum layer weights
        # Shape: (n_layers, n_qubits)  e.g. (2, 16) = 32 parameters
        # Initialized randomly -> gradient descent will optimize these
        weight_shape = get_weight_shape(n_layers, n_qubits)
        self.weights = nn.Parameter(
            torch.randn(*weight_shape, dtype=torch.float32) * 0.1
        )
        # Small random init (x0.1) avoids saturating the rotation gates
        # at the start of training, which helps gradients flow better.

        # Classical output layer
        # Takes the 16 quantum expectation values -> 10 digit class logits
        # Parameter count: 16x10 + 10 = 170
        self.classical_layer = nn.Linear(n_qubits, 10)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: image -> predicted class logits.

        Parameters
        ----------
        image : torch.Tensor  shape (1, 28, 28)  -> a single MNIST image

        Returns
        -------
        torch.Tensor  shape (10,)  -> raw logits for each digit class
        """

        # Step 1: Classical data encoding
        # Reduce 784 pixels -> 16 rotation angles
        if self.use_variance_encoding and self.top_indices is not None:
            from src.encoding import encode_data_variance
            angles = encode_data_variance(image, self.top_indices)
        else:
            angles = encode_data(image, self.n_qubits)
        # angles shape: (n_qubits,)  values in [0, pi]

        # Step 2: Quantum circuit forward pass
        # Run the variational circuit, get 16 expectation values
        q_out = quantum_circuit(angles, self.weights)
        # q_out is a list of 16 tensors, each a scalar in [-1, +1]

        # Step 3: Convert quantum output to tensor
        q_tensor = torch.stack(q_out)
        # shape: (n_qubits,)  =  (16,)

        # Step 4: Classical linear layer
        # Map 16 quantum features -> 10 class logits
        logits = self.classical_layer(q_tensor)
        # shape: (10,)

        return logits


# ---------------------------------------------------------------------------
# Classical Baseline Model  (Post 08 -> Benchmark)
# ---------------------------------------------------------------------------

class ClassicalBaseline(nn.Module):
    """
    A simple 2-layer classical neural network for fair comparison.

    Receives the SAME 16 features as the quantum model (not the full image),
    ensuring the only difference is the model architecture, not the input.

    Architecture:
      16 features -> Linear(16->32) -> ReLU -> Linear(32->10) -> 10 logits

    Parameter count:
      16x32 + 32 + 32x10 + 10  =  874 parameters

    Why this is a fair comparison:
      Both models receive 16 identical features.
      The quantum model has 32 quantum + 170 classical = 202 parameters.
      The classical model has 874 parameters.
      The classical model has more parameters but the comparison isolates
      the architectural difference (quantum vs classical processing).
    """

    def __init__(self, n_features: int = 16, n_hidden: int = 32,
                 n_classes: int = 10):
        super().__init__()

        # Hidden layer: 16 -> 32 with ReLU non-linearity
        self.hidden = nn.Linear(n_features, n_hidden)

        # Output layer: 32 -> 10 class logits
        self.output = nn.Linear(n_hidden, n_classes)

        # ReLU activation for non-linearity between layers
        self.relu = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : torch.Tensor  shape (16,) -> same 16 features as quantum model

        Returns
        -------
        torch.Tensor  shape (10,) -> class logits
        """
        x = self.relu(self.hidden(features))   # 16 -> 32 with ReLU
        return self.output(x)                  # 32 -> 10