# Phase 1 — Quantum Machine Learning (QML)

Goal: establish the **quantum + AI** cross-disciplinary identity by implementing core QML
algorithms from scratch on top of PennyLane / Qiskit, then reproducing a recent VQE / QML paper.

## Sub-projects

| # | Project | Status | Key skill |
|---|---------|--------|-----------|
| 1.1 | VQE for H₂ / LiH ground state | TODO | Variational circuits, Hamiltonian construction |
| 1.2 | Variational Quantum Classifier (Iris / circles) | TODO | Data encoding, hybrid cost |
| 1.3 | Quantum Kernel SVM vs classical RBF | TODO | Quantum kernels, kernel alignment |
| 1.4 | Quantum GAN (qGAN) on a simple distribution | TODO | Adversarial training, generator/discriminator |
| 1.5 | Reproduce one VQE / QML paper (selected) | TODO | Literature reading, figure replication |

## Running

```powershell
# from repo root, using the conda env that already has PyTorch+CUDA
conda activate pytorch_env
python 01-quantum-ml/vqe_h2.py
```

## Notes

- RTX 3050 (4 GB) is more than enough: QML workloads here use state-vector CPU simulators
  (`lightning.qubit`) for ≤ 20 qubits.
- Prefer PennyLane's `lightning.qubit` device (C++ backend) over `default.qubit` for speed.
