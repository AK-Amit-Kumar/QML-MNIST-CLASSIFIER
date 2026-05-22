"""
train.py
========
QML Series 2 | Post 07 & 08 -> Training Loop & Evaluation

Provides:
  train_model()    -> universal training loop (works for both quantum and classical)
  evaluate_model() -> accuracy evaluation on a test set
  plot_results()   -> visualization of loss curves and accuracy bars

The same train_model() function is used for both the quantum and the
classical baseline in Post 08 to ensure a perfectly fair benchmark.
"""

import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from src.encoding import encode_data


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_model(model: nn.Module,
                train_loader,
                n_epochs: int = 10,
                lr: float = 0.01,
                use_lr_schedule: bool = False,
                model_type: str = "quantum",
                n_qubits: int = 16,
                use_variance_encoding: bool = False,
                top_indices: torch.Tensor = None) -> dict:
    """
    Universal training loop -> works identically for quantum and classical models.

    Parameters
    ----------
    model                : nn.Module    -> HybridQMLModel or ClassicalBaseline
    train_loader         : DataLoader   -> training data
    n_epochs             : int          -> number of training epochs
    lr                   : float        -> initial learning rate
    use_lr_schedule      : bool         -> if True, decay LR every 3 epochs by 0.5
    model_type           : str          -> "quantum" or "classical"
    n_qubits             : int          -> features to extract (quantum only)
    use_variance_encoding: bool         -> use variance-based encoding (Post 09)
    top_indices          : Tensor|None  -> pre-computed pixel indices (Post 09)

    Returns
    -------
    dict with keys "losses" (list of per-epoch avg loss) and "time" (seconds)

    How it works
    ------------
    For each epoch:
      1. optimizer.zero_grad()   -> clear gradients from the previous step
      2. model(input)            -> forward pass (quantum: runs circuit + classical layer)
      3. loss_fn(pred, target)   -> compute cross-entropy loss
      4. loss.backward()         -> compute gradients
                                    * Classical layer: standard autograd
                                    * Quantum circuit: parameter-shift rule (Post 04)
      5. optimizer.step()        -> update ALL parameters (quantum + classical)

    Cross-Entropy Loss
    ------------------
    nn.CrossEntropyLoss is the standard choice for multi-class classification.
    It combines Softmax + Negative Log-Likelihood in one numerically stable step.
    It expects raw logits (not softmax probabilities) as input.

    Adam Optimizer
    --------------
    Adam adapts the learning rate per parameter, which helps navigate the
    irregular gradient landscape of quantum circuits. Fixed LR = 0.01 is the
    starting point (Post 07). A StepLR schedule is used in Post 09.
    """

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Optional: learning rate schedule (Post 09 optimization)
    # Decays LR by x0.5 every 3 epochs: 0.01 -> 0.005 -> 0.0025 -> 0.00125
    if use_lr_schedule:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.5
        )

    epoch_losses = []
    start_time   = time.time()

    model.train()   # enable training mode (activates dropout etc. if any)

    for epoch in range(n_epochs):
        total_loss   = 0.0
        batch_count  = 0

        for images, labels in train_loader:
            optimizer.zero_grad()   # ALWAYS clear gradients before forward pass

            batch_loss = 0.0

            # Process each image individually
            # Quantum circuits in PennyLane-Torch work best one sample at a time
            # for our circuit size; batching adds complexity without much gain here
            for image, label in zip(images, labels):

                # Get model prediction
                if model_type == "quantum":
                    # HybridQMLModel handles encoding internally
                    logits = model(image)
                else:
                    # ClassicalBaseline: manually encode to the same 16 features
                    if use_variance_encoding and top_indices is not None:
                        from src.encoding import encode_data_variance
                        features = encode_data_variance(image, top_indices)
                    else:
                        features = encode_data(image, n_qubits)
                    logits = model(features)

                # logits shape: (10,) -> need (1, 10) for loss function
                loss = loss_fn(logits.unsqueeze(0),
                               label.unsqueeze(0))
                batch_loss += loss

            # Average loss over the batch, then backprop
            batch_loss = batch_loss / len(images)
            batch_loss.backward()   # compute gradients (param-shift for quantum)
            optimizer.step()        # update weights

            total_loss  += batch_loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count
        epoch_losses.append(avg_loss)

        # Step the LR scheduler if enabled
        if use_lr_schedule:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1:02d}/{n_epochs}  |  "
                  f"Loss: {avg_loss:.4f}  |  LR: {current_lr:.6f}")
        else:
            print(f"  Epoch {epoch+1:02d}/{n_epochs}  |  Loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"\n  Training complete in {elapsed:.1f}s")

    return {"losses": epoch_losses, "time": elapsed}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model: nn.Module,
                   test_loader,
                   model_type: str = "quantum",
                   n_qubits: int = 16,
                   use_variance_encoding: bool = False,
                   top_indices: torch.Tensor = None) -> float:
    """
    Evaluate accuracy on the test set.

    Parameters
    ----------
    model      : nn.Module  -> trained model
    test_loader: DataLoader -> test data
    model_type : str        -> "quantum" or "classical"
    n_qubits   : int        -> features to extract

    Returns
    -------
    float -> accuracy in [0, 1]

    How it works
    ------------
    model.eval()       -> disables training-only behaviours (dropout, batchnorm)
    torch.no_grad()    -> skips gradient tracking (saves memory, speeds up eval)
    torch.argmax()     -> picks the class with the highest logit as the prediction
    """

    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in test_loader:
            for image, label in zip(images, labels):

                if model_type == "quantum":
                    logits = model(image)
                else:
                    if use_variance_encoding and top_indices is not None:
                        from src.encoding import encode_data_variance
                        features = encode_data_variance(image, top_indices)
                    else:
                        features = encode_data(image, n_qubits)
                    logits = model(features)

                predicted = torch.argmax(logits).item()
                actual    = label.item()

                if predicted == actual:
                    correct += 1
                total += 1

    accuracy = correct / total
    print(f"  Accuracy: {correct}/{total} = {accuracy*100:.1f}%")
    return accuracy


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_loss_curves(results: dict, save_path: str = "results/loss_curves.png"):
    """
    Plot training loss curves for one or more models.

    Parameters
    ----------
    results   : dict -> keys are model names, values are training result dicts
                        e.g. {"Quantum": q_results, "Classical": c_results}
    save_path : str  -> where to save the figure
    """
    plt.figure(figsize=(9, 5))

    colors = ["#6366f1", "#10b981", "#f59e0b", "#ef4444"]
    for (name, res), color in zip(results.items(), colors):
        plt.plot(
            range(1, len(res["losses"]) + 1),
            res["losses"],
            marker="o", markersize=4,
            label=f"{name}  (final loss: {res['losses'][-1]:.3f})",
            color=color, linewidth=2
        )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.title("Training Loss: Quantum vs Classical", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Loss curves saved -> {save_path}")


def plot_accuracy_bar(accuracies: dict,
                      save_path: str = "results/accuracy_comparison.png"):
    """
    Bar chart comparing test accuracies of different model configurations.

    Parameters
    ----------
    accuracies : dict -> keys are model names, values are accuracy floats
    save_path  : str  -> where to save the figure
    """
    names  = list(accuracies.keys())
    values = [v * 100 for v in accuracies.values()]   # convert to %

    colors = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, values, color=colors[:len(names)], edgecolor="white",
                   linewidth=1.2)

    # Add value labels on top of each bar
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.8,
                 f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")

    # Random chance reference line
    plt.axhline(y=10, color="red", linestyle="--",
                label="Random chance (10%)", alpha=0.7)

    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.title("Accuracy Comparison: Quantum vs Classical", fontsize=14)
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Accuracy chart saved -> {save_path}")