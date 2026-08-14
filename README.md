# AI4Science × Quantum ML Portfolio

Personal research portfolio targeting PhD applications in **AI4Science**, **quantum
information**, and the **quantum + AI** intersection. Author: a Physics undergraduate at Sun
Yat-sen University (中山大学).

This repository hosts a staged, reproducible portfolio of projects — from quantum machine
learning foundations through neural quantum states to equivariant neural network potentials —
designed to build the exact cross-disciplinary skill set that top programs (HYPSM, UCB, ETH,
HK top-3, SG top-2, TUM, Caltech, Columbia, Penn) look for.

## Environment

A single conda environment is reused across all phases (PyTorch + CUDA already configured):

```powershell
conda activate pytorch_env
```

| Stack | Package | Version | Purpose |
|-------|---------|---------|---------|
| DL (GPU) | torch / torchvision / torchaudio | 2.5.1 | Neural nets, PINNs, NQS (PyTorch backend) |
| QML | pennylane + pennylane-lightning + pennylane-qiskit | 0.42.x | Variational circuits, VQE, quantum kernels |
| Quantum | qiskit + qiskit-aer + qiskit-ibm-runtime | 1.2.x | Circuit sim, IBM runtime |
| Many-body ML | netket | 3.17.x | Neural quantum states (JAX backend) |
| Autodiff (CPU) | jax + optax + flax | 0.5.x | Differentiable physics, NQS |
| Scientific | numpy / scipy / matplotlib / scikit-learn / einops | — | Core numerics, plotting, ML utils |

> GPU: NVIDIA RTX 3050 Laptop (4 GB). QML uses CPU state-vector simulators; PINN/NQS training
> runs on GPU. See [`ROADMAP.md`](./ROADMAP.md) for the per-phase package plan.

> Deferred (install when reaching that phase): `e3nn`, `ase`, `pyscf`, `torch-geometric`
> (Phase 3); `quimb`/`cotengra` currently conflict with PennyLane's `autoray` pin — reinstall
> when needed for tensor-network work.

## Portfolio structure

```
01-quantum-ml/      Phase 1 — Quantum Machine Learning (VQE, QML classifiers, quantum kernels)
02-pinn-nqs/        Phase 2 — Physics-Informed NN & Neural Quantum States
03-equivariant-nn/  Phase 3 — E(3)-equivariant NN potentials & molecule generation
04-paper-repro/     Phase 4 — Paper reproduction + original extension (application centerpiece)
```

Each sub-folder has its own README with a project table and run instructions.

## Recommended GitHub repositories to study & contribute

Curated by sub-direction. Star, read the source, run their demos, then attempt a small
reproduction or issue fix — this is the fastest path to a credible contribution record.

### Quantum ML & quantum computing
- **PennyLaneAI/pennylane** — the canonical QML framework. Start with their `qml/demos`.
- **Qiskit/qiskit** + **Qiskit/qiskit-aer** — IBM stack; `qiskit-textbook` for foundations.
- **netket/netket** — ML for quantum many-body physics (Becca group, EPFL). Excellent for NQS.
- **quantumlib/Cirq** — Google's quantum framework.
- **tensorflow/quantum** — Google's TFQ (QML on TF).
- **unitaryfund/mitiq** — quantum error mitigation; beginner-friendly issues.

### Neural quantum states & quantum Monte Carlo
- **google-deepmind/ferminet** — neural-network wavefunctions for electrons (Spencer et al.).
- **deepqmc/deepqmc** — deep QMC for molecules (PauliNet lineage).

### AI4Science — equivariant NN & molecules
- **e3nn/e3nn** — Euclidean equivariant NN library; *the* entry point for Phase 3.
- **mir-group/nequip** — E(3)-equivariant neural network potentials (Kozinsky group, Harvard).
- **ACEsuit/mace** — highly accurate equivariant MLP (Batatia, Cambridge) — very hot.
- **jax-md/jax-md** — differentiable molecular dynamics in JAX.

### Physics-informed neural networks
- **maziarraissi/PINNs** — original PINNs code by Raissi.
- **lab-cosmo** PINN tutorial collections.

### Quantum chemistry / quantum info (supporting)
- **pyscf/pyscf** — quantum chemistry (needed for molecular Hamiltonians in VQE).
- **qutip/qutip** — quantum optics / open systems toolbox.
- **jcmgray/quimb** — tensor networks (mind the autoray conflict noted above).

## Quick start

```powershell
conda activate pytorch_env
python 01-quantum-ml/vqe_h2.py   # runs VQE for H2 ground state
```

## Roadmap

See [`ROADMAP.md`](./ROADMAP.md) for the detailed four-phase plan: skills, papers, projects,
timeline, and PhD application strategy.

## License

MIT (see existing LICENSE in repo).
