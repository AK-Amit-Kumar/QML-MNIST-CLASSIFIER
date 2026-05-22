"""
circuit.py
==========
QML Series 2 | Post 06 -> Variational Quantum Circuit (Ansatz)

Defines the parameterized quantum circuit that acts as the
"feature extractor" in our hybrid quantum-classical model.

Architecture (matches Post 03 & 04 theory):
  1. Encoding layer    -> AngleEmbedding maps data to quantum state
  2. Variational layer -> BasicEntanglerLayers (learnable parameters)
  3. Measurement layer -> PauliZ expectation values on every qubit
"""

import pennylane as qml
from pennylane import numpy as np
import torch


# ---------------------------------------------------------------------------
# Device: lightning.qubit is the fast C++ simulator (Post 05)
# ---------------------------------------------------------------------------

N_QUBITS = 16    # one qubit per encoded feature
N_LAYERS = 2     # number of variational layers (shallow = NISQ-friendly)

dev = qml.device("lightning.qubit", wires=N_QUBITS)


# ---------------------------------------------------------------------------
# The Quantum Circuit (QNode)
# ---------------------------------------------------------------------------

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_circuit(inputs: torch.Tensor, weights: torch.Tensor):
    """
    A variational quantum circuit for hybrid QML classification.

    Parameters
    ----------
    inputs  : torch.Tensor shape (N_QUBITS,)
              Rotation angles from the encoding layer, one per qubit.
    weights : torch.Tensor shape (N_LAYERS, N_QUBITS)
              Trainable parameters, optimized by gradient descent.

    Returns
    -------
    List of N_QUBITS expectation values, each in [-1, +1].

    Step-by-step explanation
    ------------------------
    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
        Turns this Python function into a "quantum node".
        interface="torch"  -> gradients flow into PyTorch autograd.
        diff_method="parameter-shift" -> uses the parameter-shift rule
        (discussed in Post 04) to compute gradients through the circuit.

    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
        DATA ENCODING LAYER.
        Applies RY(angle) to each qubit using the pre-computed angles.
        RY(theta)|0> rotates the qubit Bloch-sphere state by angle theta
        around the Y-axis, encoding our classical data into quantum state.

    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
        VARIATIONAL LAYER (the learning engine).
        For each layer:
            * Applies RX(weight) rotation to every qubit.
            * Applies CNOT gates in a ring pattern to entangle qubits.
        This is repeated N_LAYERS times.
        The weights are what gradient descent will optimize.

    [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
        MEASUREMENT LAYER.
        Measures the expectation value of the Pauli-Z operator on each qubit.
        Each measurement returns a real number in [-1, +1].
        These N_QUBITS numbers are the output of the quantum circuit,
        passed to the classical layer for final classification.
    """

    # 1. Data Encoding
    # Apply RY(input[i]) to qubit i for each of the 16 qubits
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")

    # 2. Variational Layers
    # weights shape: (N_LAYERS, N_QUBITS)
    # Each layer: RX rotations + entangling CNOTs
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))

    # 3. Measurement
    # Return expectation value of Z on every qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# ---------------------------------------------------------------------------
# Utility: get the correct weight tensor shape
# ---------------------------------------------------------------------------

def get_weight_shape(n_layers: int = N_LAYERS,
                     n_qubits: int = N_QUBITS) -> tuple:
    """
    Returns the shape (n_layers, n_qubits) for the weights tensor.

    Total trainable quantum parameters = n_layers x n_qubits
    With defaults: 2 x 16 = 32 parameters.

    For comparison, a simple classical NN for the same task
    would typically have thousands of parameters.
    """
    return (n_layers, n_qubits)


# ---------------------------------------------------------------------------
# Utility: draw the circuit (helpful for understanding / debugging)
# ---------------------------------------------------------------------------

def draw_circuit(n_layers: int = N_LAYERS, n_qubits: int = N_QUBITS):
    """
    Print a text diagram of the circuit structure.
    Useful for verifying the ansatz design visually.
    """
    sample_inputs  = np.zeros(n_qubits)
    sample_weights = np.zeros((n_layers, n_qubits))

    print(qml.draw(quantum_circuit)(
        torch.tensor(sample_inputs, dtype=torch.float32),
        torch.tensor(sample_weights, dtype=torch.float32)
    ))