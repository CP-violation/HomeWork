import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1) 读取最优基因
# ===============================
gene = np.load("car_track/best_pp_params.npy")   # ← 需确保路径正确
assert gene.shape[0] == 45, "基因长度应为 45，请检查文件是否正确"

# ===============================
# 2) 解析三角 MF 参数
#   前 18 基因 → 误差 e MF & de MF
#   每 3 个为一组 (a,b,c) → 经 sort 保证 a<=b<=c
# ===============================
mf_values = gene[:18].reshape(6, 3)
mf_values = np.sort(mf_values, axis=1)   # 确保 a,b,c 单调
mf_e  = mf_values[:3]   # 误差 e MF
mf_de = mf_values[3:]   # 导数 de MF

print("MF - e:\n", mf_e)
print("MF - de:\n", mf_de)

# ===============================
# 3) 三角隶属度函数
# ===============================
def tri_mf(x, a, b, c):
    return np.maximum(np.minimum((x-a)/(b-a+1e-9), (c-x)/(c-b+1e-9)), 0.0)

# ===============================
# 4) 绘制三角 MF 图
# ===============================
x = np.linspace(0, 1, 400)

fig, ax = plt.subplots(1, 2, figsize=(10,4))

# ---- 误差 e ----
for i,(a,b,c) in enumerate(mf_e):
    ax[0].plot(x, tri_mf(x,a,b,c), label=f"E{i+1} [{a:.2f},{b:.2f},{c:.2f}]")
ax[0].set_title("Triangular MF for Error e")
ax[0].set_xlabel("Normalized Error")
ax[0].set_ylabel("Membership Degree")
ax[0].legend()

# 强制第一个函数左端点 ≤ 0，最后一个函数右端点 ≥ 1

# ---- 导数 de ----


a=0.00; b=0.41; c=0.55
ax[1].plot(x, tri_mf(x,a,b,c), label=f"dE{1} [{a:.2f}, {b:.2f}, {c:.2f}]")
a=0.29; b=0.51; c=0.59
ax[1].plot(x, tri_mf(x,a,b,c), label=f"dE{2} [{a:.2f},{b:.2f},{c:.2f}]")
a=0.41; b=0.71; c=1.00
ax[1].plot(x, tri_mf(x,a,b,c), label=f"dE{3} [{a:.2f}, {b:.2f}, {c:.2f}]")
ax[1].set_title("Triangular MF for Error Derivative de")
ax[1].set_xlabel("Normalized dError")
ax[1].set_ylabel("Membership Degree")
ax[1].legend()

plt.tight_layout()
plt.show()


print(gene)
