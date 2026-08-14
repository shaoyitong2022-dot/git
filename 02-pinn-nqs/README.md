# Phase 2 — Physics-Informed Neural Networks & Neural Quantum States

Goal: demonstrate **AI4Science** capability by solving physics PDEs/eigen-problems with neural
networks, and learning quantum many-body ground states with neural-network wavefunctions.

## Sub-projects

| # | Project | Status | Key skill |
|---|---------|--------|-----------|
| 2.1 | PINN for 1D Schrödinger (HO, finite well) | TODO | Autograd, collocation loss |
| 2.2 | PINN for 2D heat / diffusion eq. | TODO | PDE conditioning |
| 2.3 | Neural Quantum State (RBM) for transverse-field Ising | TODO | Carleo–Troyer, NetKet |
| 2.4 | NQS for Heisenberg chain, energy + correlation benchmarks | TODO | Variational MC, sampling |
| 2.5 | Reproduce a recent AI4Sci / NQS paper | TODO | |

## Running

```powershell
conda activate pytorch_env
python 02-pinn-nqs/pinn_schrodinger.py
```
