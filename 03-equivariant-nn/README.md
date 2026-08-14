# Phase 3 — Equivariant Neural Networks & AI4Science (Molecules)

Goal: enter the highest-impact AI4Sci sub-field — **E(3)-equivariant neural network potentials
and generative models for molecules**. This is the area with the most NeurIPS/ICML/Nature papers
and the most direct industry path (DeepMind, Microsoft Research, drug discovery).

> Note: install when starting this phase — not part of the base env to keep it light:
> `pip install e3nn ase pyscf torch-geometric`

## Sub-projects

| # | Project | Status | Key skill |
|---|---------|--------|-----------|
| 3.1 | e3nn tutorial: equivariant conv on a small molecule | TODO | Irreps, SO(3) equivariance |
| 3.2 | Train NequIP-style potential on MD17 (ethanol, aspirin) | TODO | MLIP, training loop |
| 3.3 | Equivariant diffusion for small molecule generation (EDM-lite) | TODO | Diffusion, equivariant score |
| 3.4 | Bench MACE on a custom dataset | TODO | ACE, higher-body correlations |

## Reference repos

- e3nn/e3nn, mir-group/nequip, ACEsuit/mace, jax-md/jax-md
