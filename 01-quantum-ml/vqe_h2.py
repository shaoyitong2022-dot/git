"""
Phase 1 起步脚本: 变分量子本征求解器 (VQE) 求解 H2 分子基态能量。

这是量子机器学习 / 量子化学最经典的入门实验。目标是:
1. 构造 H2 (STO-3G, R=0.74 A) 简化到 2 量子比特的有效哈密顿量
2. 设计参数化量子电路 (ansatz) 作为试探波函数
3. 用经典优化器最小化 <H> 得到基态能量
4. 与精确对角化结果对比

环境: conda activate pytorch_env
运行:  python 01-quantum-ml/vqe_h2.py
"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp  # autograd 后端的 numpy, 支持 requires_grad
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. H2 简化哈密顿量 (2 量子比特形式, Kandala et al. 2017 / Whitfield 形式)
#    H = g0*I + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*X0X1 + g5*Y0Y1
#    这是 4 量子比特 H2 在 STO-3G 下利用电子数/自旋对称性 (Bravyi-Kitaev +
#    两个守恒量) 压缩到 2 量子比特后的形式, 对应 R=0.74 A (平衡键长)。
#    基态能量约 -1.857 Hartree。
# ---------------------------------------------------------------------------
g0, g1, g2, g3, g4, g5 = -0.4804, 0.3435, -0.4347, 0.5716, 0.0910, 0.0910

HAMILTONIAN = (
    g0 * qml.I(0)
    + g1 * qml.Z(0)
    + g2 * qml.Z(1)
    + g3 * qml.Z(0) @ qml.Z(1)
    + g4 * qml.X(0) @ qml.X(1)
    + g5 * qml.Y(0) @ qml.Y(1)
)

# ---------------------------------------------------------------------------
# 2. 参数化量子电路 (ansatz): 硬件高效型, 2 个 RY + 1 个 CNOT
#    单参数 RY-CNOT 只能张成 {|00>,|11>} 子空间, 够不到 H2 真正基态
#    (基态含 |01>/|10> 分量)。加一个 RY 在 q1 上即覆盖完整 2-qubit
#    Hilbert 空间, 足以表示任意本征态。
# ---------------------------------------------------------------------------
N_QUBITS = 2
dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev)
def ansatz(theta):
    # theta = [θ0, θ1]
    qml.RY(theta[0], wires=0)
    qml.RY(theta[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(HAMILTONIAN)


def cost(theta):
    """经典代价函数 = 在 ansatz 状态下的能量期望。"""
    return ansatz(theta)


# ---------------------------------------------------------------------------
# 3. 精确解 (用来对比): 直接对 4x4 哈密顿矩阵做对角化
# ---------------------------------------------------------------------------
H_matrix = qml.matrix(HAMILTONIAN, wire_order=[0, 1])
exact_eigs = np.linalg.eigvalsh(H_matrix)
E_exact = exact_eigs[0]

# ---------------------------------------------------------------------------
# 4. 经典优化: 2D 多起点网格 + L-BFGS-B (VQE 标准做法)
#    变分算法对初值敏感, 工程做法: 在参数空间撒网格, 各自跑到收敛, 取最低能量。
# ---------------------------------------------------------------------------
from scipy.optimize import minimize

grid = np.linspace(-np.pi, np.pi, 5, endpoint=False)
history = []
best_E, best_theta = np.inf, None
print("2D 多起点优化 (L-BFGS-B), 25 个起点:")
n_start = 0
for th0 in grid:
    for th1 in grid:
        n_start += 1
        res = minimize(lambda x: float(cost(pnp.array(x))), x0=[th0, th1],
                       method="L-BFGS-B")  # 数值梯度, 对 2 参数小问题足够
        x_opt = np.array(res.x, dtype=float)
        E_i = float(cost(pnp.array(x_opt)))
        if E_i < best_E:
            best_E, best_theta = E_i, x_opt
            history.append(E_i)
            print(f"  start {n_start:2d} θ0={th0:+.2f},θ1={th1:+.2f} -> "
                  f"θ*=({x_opt[0]:+.3f},{x_opt[1]:+.3f}) E={E_i:+.6f}  *new best")

E_vqe = best_E
theta_opt = best_theta
print(f"\n全局最优: θ*=({theta_opt[0]:+.4f},{theta_opt[1]:+.4f})  E_VQE={E_vqe:+.6f} Ha")

# ---------------------------------------------------------------------------
# 5. 输出 + 绘图
# ---------------------------------------------------------------------------
print("\n========== 结果 ==========")
print(f"VQE 能量:      {E_vqe:+.6f} Hartree")
print(f"精确对角化:    {E_exact:+.6f} Hartree")
print(f"绝对误差:      {abs(E_vqe - E_exact):.2e} Hartree")
print(f"(实验值参考:   -1.857 Ha, R=0.74 A)")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# 左: 多起点找到的全局最优能量下降轨迹
axes[0].plot(range(len(history)), history, "o-", ms=5, color="C0",
             label="best $\\langle H \\rangle$ so far")
axes[0].axhline(E_exact, color="r", ls="--", label=f"Exact = {E_exact:.4f} Ha")
axes[0].axhline(-1.857, color="g", ls=":", label="Experiment -1.857 Ha")
axes[0].set_xlabel("Improvement event index (multi-start)")
axes[0].set_ylabel("Energy (Hartree)")
axes[0].set_title("VQE multi-start best energy for H$_2$ (STO-3G, R=0.74 Å)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 右: 能量切片 —— 固定 θ0=θ0*, 扫描 θ1 (展示极小位置)
th1s = np.linspace(-np.pi, np.pi, 200)
slice_energies = [float(ansatz(pnp.array([theta_opt[0], t]))) for t in th1s]
axes[1].plot(th1s, slice_energies, "-", color="C0")
axes[1].plot(theta_opt[1], E_vqe, "r*", ms=15,
             label=f"VQE optimum θ1*={theta_opt[1]:+.3f}")
axes[1].axhline(E_exact, color="g", ls=":", label="Exact ground energy")
axes[1].set_xlabel("Ansatz parameter $\\theta_1$ (with $\\theta_0=\\theta_0^*$ fixed)")
axes[1].set_ylabel("Energy (Hartree)")
axes[1].set_title("Energy landscape slice (showing the minimum)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
out_path = "01-quantum-ml/vqe_h2_result.png"
plt.savefig(out_path, dpi=120)
print(f"\n图已保存: {out_path}")
print("下一步: 修改 g0..g5 为不同键长 R 对应的系数, 扫描出完整势能曲线 E(R)。")
