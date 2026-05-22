<div align="center">

# Quantum Machine Learning for Developers

### _Can a quantum computer learn to read handwriting?_

**A honest, hands-on answer : built across 9 articles, measured with real numbers.**

<br/>

[![PennyLane](https://img.shields.io/badge/PennyLane-0.45.0-6366f1?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PC9zdmc+)](https://pennylane.ai)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12.0-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-f37626?style=for-the-badge&logo=jupyter&logoColor=white)](notebooks/QML_Complete_Project.ipynb)
[![License](https://img.shields.io/badge/License-MIT-10b981?style=for-the-badge)](LICENSE)

<br/>

> _"Most QML content blurs the line between theoretical advantage and what actually runs on hardware today._
> _This project does not."_
>
> - Amit Kumar, QML Series 2

<br/>

**[ Open the Notebook](notebooks/QML_Complete_Project.ipynb)** &nbsp;|&nbsp;
**[ Read the Series](#-the-series-that-built-this)** &nbsp;|&nbsp;
**[ Jump to Results](#-honest-results)** &nbsp;|&nbsp;
**[ Quick Start](#-quick-start)**

</div>

---

## What Is This, Really?

This is **not** another "quantum computing is the future" hype piece.

This is the code, the training loop, the benchmark numbers, and the honest lessons from building a **hybrid quantum-classical image classifier from scratch** - the kind of project most QML tutorials skip straight past.

By the end of this README you will understand:

- What a variational quantum circuit actually looks like as running Python code
- Why the classical model wins the benchmark (and why that result is still interesting)
- What the **single most impactful optimization** turned out to be (it was not the circuit)
- What you need to run this yourself in under 10 minutes

---

## The Series That Built This

This repository is the living codebase for **QML Series 2**, a 9-post LinkedIn series building quantum ML from first principles. Each post added one piece. This repo holds all of them.

| #   | Post                                           | What Gets Built                 | Link                                                                                                            |
| --- | ---------------------------------------------- | ------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| 01  | What is QML and why does it matter?            | Theory foundation               | [Read >](https://www.linkedin.com/pulse/quantum-machine-learning-developers-what-qml-why-does-amit-kumar-zyymc) |
| 02  | Where quantum beats classical ML               | Honest capability map           | [Read >](https://www.linkedin.com/pulse/where-quantum-beats-classical-ml-does-amit-kumar-bxb1c)                 |
| 03  | Quantum Neural Networks explained              | QNN architecture                | [Read >](https://www.linkedin.com/pulse/quantum-neural-networks-explained-amit-kumar-c3j5c)                     |
| 04  | Variational Quantum Circuits                   | The parameter-shift rule        | [Read >](https://www.linkedin.com/pulse/variational-quantum-circuits-engine-qml-amit-kumar-bd6kc)               |
| 05  | **Setting Up the QML Environment**             | `PennyLane + PyTorch install`   | [Read >](https://www.linkedin.com/pulse/setting-up-your-qml-environment-amit-kumar-izzdc)                       |
| 06  | **Building Your First Quantum Classifier**     | `encoding.py` + `circuit.py`    | [Read >](https://www.linkedin.com/pulse/building-your-first-quantum-classifier-amit-kumar-9xf8c)                |
| 07  | **Training a Quantum Neural Network**          | `model.py` + `train.py`         | [Read >](https://www.linkedin.com/pulse/training-quantum-neural-network-amit-kumar-eyffc)                       |
| 08  | **Quantum vs Classical: The Honest Benchmark** | Fair comparison, real numbers   | [Read >](https://www.linkedin.com/pulse/quantum-vs-classical-ml-honest-benchmark-amit-kumar-qcwsc)              |
| 09  | **Optimizing Your Quantum Model**              | 4 levers tested, 1 clear winner | [Read >](https://www.linkedin.com/pulse/optimizing-your-quantum-model-amit-kumar-1xn7c)                         |

> Posts 01-04 are theory. **Posts 05-09 are this repository.**

---

## What We Built

A **hybrid quantum-classical neural network** that classifies handwritten digits (MNIST 0-9).

The full data pipeline looks like this:

```
MNIST Image  (28 x 28 = 784 pixels)
        |
        v
[ Classical Pre-processing ]
  encode_data()
  784 pixels -> 16 rotation angles in [0, pi]
        |
        v
[ Variational Quantum Circuit ]   <-- the interesting part
  AngleEmbedding    : 16 angles -> 16 qubit states
  BasicEntanglerLayers x2 : RX rotations + CNOT ring (32 learnable params)
  PauliZ measurements : 16 expectation values in [-1, +1]
        |
        v
[ Classical Output Layer ]
  nn.Linear(16 -> 10)  :  170 parameters
        |
        v
  Predicted digit  (0 through 9)
```

**Total trainable parameters: 32 (quantum) + 170 (classical) = 202**

For context, a classical CNN on the same task uses hundreds of thousands of parameters. The quantum circuit is extraordinarily parameter-efficient. Whether that efficiency translates to competitive accuracy is the question this project answers.

---

## Project Structure

```
qml-project/
|
+-- notebooks/
|   +-- QML_Complete_Project.ipynb   <- START HERE (25 cells, fully commented)
|
+-- src/
|   +-- encoding.py      <- angle encoding: strided + variance-based strategies
|   +-- circuit.py       <- 16-qubit variational quantum circuit (ansatz)
|   +-- model.py         <- HybridQMLModel + ClassicalBaseline
|   +-- train.py         <- training loop, evaluation, loss/accuracy plots
|   +-- data_loader.py   <- MNIST dataset loading and batching
|
+-- data/                <- auto-created by torchvision on first run
+-- results/             <- training curves, accuracy charts, saved models
+-- requirements.txt
+-- README.md
```

> **The notebook is the primary entry point.** The `src/` files are the same code in modular form for reuse.

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/qml-mnist-classifier.git
cd qml-mnist-classifier
```

### 2. Create virtual environment

```bash
# Mac / Linux
python -m venv qml-env
source qml-env/bin/activate

# Windows (PowerShell)
python -m venv qml-env
.\qml-env\Scripts\Activate.ps1
```

> **Windows users:** If you hit a long-path error during installation, move the project to `C:\qml\` - this is a known Windows 260-character path limit issue.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install pennylane pennylane-lightning torch torchvision numpy matplotlib scikit-learn jupyter
```

### 4. Launch the notebook

```bash
jupyter notebook notebooks/QML_Complete_Project.ipynb
```

Run cells **top to bottom**. Skip Cell 01 (installation) since you just installed everything.

> **Expected training time:** 3-8 minutes for the quantum model on CPU. This is intentional and explained in the notebook - it is one of the most important practical lessons of the series.

---

## The Key Ideas (Without the Jargon)

### Why 16 qubits and not 784?

MNIST images have 784 pixels. Each qubit can encode one feature. Running 784 qubits on any current hardware (real or simulated) is not practical. So we reduce 784 pixels to 16 before the circuit sees anything.

This is not a shortcut. It is the honest constraint of NISQ (Noisy Intermediate-Scale Quantum) hardware today. A classical CNN does not have this limitation. The benchmark reflects this.

### What is the "parameter-shift rule"?

Classical neural networks compute gradients using backpropagation, a chain rule through differentiable operations. Quantum circuits cannot use backprop directly.

Instead, PennyLane uses the **parameter-shift rule**: to find the gradient of a circuit parameter `theta`, run the circuit twice: once with `theta + pi/2` and once with `theta - pi/2` - and take the difference divided by 2. This doubles the circuit evaluations per training step, which is why quantum training is slower.

### Why is the classical model faster?

Each quantum circuit evaluation runs on a CPU simulator. Each backward pass requires 2 circuit evaluations per quantum parameter. With 32 quantum parameters, that is 64 circuit evaluations per sample per step. A classical network does one matrix multiply. The ~80x speed difference measured in Post 08 comes directly from this.

---

## Honest Results

No cherry-picking. These are the numbers from the actual runs.

### Post 08: Baseline Benchmark

| Model              | Test Accuracy | Training Time | Parameters |
| ------------------ | :-----------: | :-----------: | :--------: |
| Quantum Classifier |     ~38%      |  ~4 minutes   |    202     |
| Classical Baseline |     ~67%      |  ~3 seconds   |    874     |
| Random Chance      |      10%      |      n/a      |    n/a     |

Both models receive the **same 16 features**. The only variable is the architecture.

**The classical model wins on every metric.** This is consistent with Post 02's theoretical analysis: image classification is not where quantum ML has a practical advantage on current hardware.

### Post 09: Optimization Results

Four levers tested. One clear winner.

| Configuration                          |  Accuracy   | Change vs Baseline |
| -------------------------------------- | :---------: | :----------------: |
| Quantum baseline (strided sampling)    |    ~38%     |        n/a         |
| + 3 variational layers                 |   ~37-40%   |     Negligible     |
| **+ Variance-based feature selection** | **~48-52%** |   **+10-14 pp**    |
| + Learning rate schedule               |   ~40-44%   |      +2-6 pp       |
| + 1000 training images                 |   ~45-50%   |      +7-12 pp      |
| **Optimized (all best choices)**       | **~52-56%** |   **+14-18 pp**    |
| Classical baseline                     |    ~67%     |     reference      |

### The most important finding

> The biggest accuracy improvement came from **better feature selection**, not from a better quantum circuit.
>
> Switching from strided pixel sampling (every 49th pixel) to **variance-based selection** (the 16 most discriminative pixels) closed more of the gap than any circuit architecture change.
>
> The quantum circuit was working hard on 16 nearly useless pixels. Give it 16 informative pixels and it performs noticeably better. Garbage in, garbage out applies to quantum models too.

---

## Tools & Stack

| Tool                                                                | Version | Role                                                     |
| ------------------------------------------------------------------- | ------- | -------------------------------------------------------- |
| [PennyLane](https://pennylane.ai)                                   | 0.45.0  | Quantum circuit construction + parameter-shift gradients |
| [pennylane-lightning](https://docs.pennylane.ai/projects/lightning) | 0.45.0  | Fast C++ simulator backend (~10x faster than default)    |
| [PyTorch](https://pytorch.org)                                      | 2.12.0  | Classical ML backbone, training loop, optimizers         |
| [torchvision](https://pytorch.org/vision)                           | 0.27.0  | MNIST dataset download and transforms                    |
| [NumPy](https://numpy.org)                                          | 2.4.6   | Numerical operations                                     |
| [Matplotlib](https://matplotlib.org)                                | 3.10.9  | Loss curves, accuracy charts, variance maps              |
| Jupyter Notebook                                                    | n/a     | Development and exploration environment                  |

---

## Inside the Notebook

The master notebook [`QML_Complete_Project.ipynb`](notebooks/QML_Complete_Project.ipynb) has **25 cells** organized into 5 sections:

```
Section 1 (Cells 01-04)  ->  Post 05: Environment setup + first circuit
Section 2 (Cells 05-09)  ->  Post 06: MNIST loading + encoding + circuit + forward pass
Section 3 (Cells 10-12)  ->  Post 07: Hybrid model + training + evaluation
Section 4 (Cells 13-16)  ->  Post 08: Classical baseline + benchmark + charts
Section 5 (Cells 17-26)  ->  Post 09: 4 optimizations + optimized model + final results
```

Every cell has:

- A markdown explanation of **what** the code does and **why**
- Line-by-line comments inside the code
- Expected outputs noted in comments

---

## Important Notes Before You Run

**Quantum training is slow by design.** The CPU simulator evaluates the circuit 64 times per sample per backward pass. For 500 images x 10 epochs, expect 3-8 minutes. This is not a bug - it is one of the key practical lessons of the series.

**This uses a simulator, not real quantum hardware.** `lightning.qubit` is a perfect noiseless simulator running locally. Real quantum hardware adds noise that would further reduce accuracy. The simulator gives us clean results for learning purposes.

**16 features is a NISQ constraint, not a design choice.** A classical CNN uses all 784 pixels through learned convolutional filters. Our circuit is bounded by the number of qubits that can be simulated efficiently on a laptop.

---

## What I Learned Building This

A few things that only became clear by writing the code:

**1. The encoding step is where most of the information is lost.**
Going from 784 pixels to 16 with strided sampling throws away more than 97% of the image. Every other part of the system could be perfect and the classifier would still be limited by this bottleneck.

**2. Circuit depth is not the lever you think it is.**
Adding a third variational layer (which should theoretically increase expressiveness) - made almost no consistent difference. The gradients are too small and the input quality is the binding constraint.

**3. Quantum optimization landscapes look different from classical ones.**
The loss curves from quantum training have more epoch-to-epoch variability than classical training. The parameter-shift rule produces noisier gradients. A decaying learning rate helps more than it would for a classical network.

**4. The parameter efficiency is genuinely remarkable.**
32 quantum parameters learning from 500 images achieving 38% accuracy on a 10-class problem. That is not a failure - it is a demonstration that variational circuits can extract signal from real-world data. They just cannot yet extract it as effectively as classical architectures designed for images.

---

## About This Project

Built by **[Amit Kumar](https://www.linkedin.com/in/amit-kumar-160767191/)** as part of the Quantum Machine Learning for Developers series.

If you found this useful:

- Star the repository
- Follow the [LinkedIn series](https://www.linkedin.com/in/amit-kumar-160767191/) for Post 10 and beyond
- Drop a comment on any of the articles - the best conversations in this series came from reader questions

---

## License

MIT License: free to use, fork, learn from, and build on.

---

<div align="center">

_Built with intellectual honesty. The numbers are the numbers, whatever they are._

**[Back to top](#-quantum-machine-learning-for-developers)**

</div>
