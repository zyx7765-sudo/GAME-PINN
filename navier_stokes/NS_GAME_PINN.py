# -*- coding: utf-8 -*-
"""
File: game_pinn_ns_stable_cosine.py
Description: GAME-PINN 稳健冲刺 10^-3 优化版 (恢复 scale=1.0 + 对称权重 + 引入余弦退火)
"""
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS,Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import scipy.io

# 1. 设置随机种子与设备
seed = 4321
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
start_time = time.time()

# 2. 加载数据
data = scipy.io.loadmat('./cylinder_nektar_wake.mat')

U_star = data['U_star']  # N x 2 x T
P_star = data['p_star']  # N x T
t_star = data['t']  # T x 1
X_star = data['X_star']  # N x 2

N = X_star.shape[0]
T = t_star.shape[0]

XX = np.tile(X_star[:, 0:1], (1, T))
YY = np.tile(X_star[:, 1:2], (1, T))
TT = np.tile(t_star, (1, N)).T

UU = U_star[:, 0, :]
VV = U_star[:, 1, :]
PP = P_star

x = XX.flatten()[:, None]
y = YY.flatten()[:, None]
t = TT.flatten()[:, None]
u = UU.flatten()[:, None]
v = VV.flatten()[:, None]
p = PP.flatten()[:, None]

# 保持 5000 个高质量初始采样点
idx = np.random.choice(N * T, 5000, replace=False)
x_train = x[idx, :]
y_train = y[idx, :]
t_train = t[idx, :]
u_train = u[idx, :]
v_train = v[idx, :]
p_train = p[idx, :]

x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=True).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=True).to(device)
t_train = torch.tensor(t_train, dtype=torch.float32, requires_grad=True).to(device)
u_train = torch.tensor(u_train, dtype=torch.float32, requires_grad=True).to(device)
v_train = torch.tensor(v_train, dtype=torch.float32, requires_grad=True).to(device)
p_train = torch.tensor(p_train, dtype=torch.float32, requires_grad=True).to(device)


# ==========================================
# 模块 A: 核心组件 (Fourier Scale 恢复为 1.0)
# ==========================================
class AdaptiveMappingNet(nn.Module):
    def __init__(self, in_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, in_dim)
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return x + 0.01 * torch.tanh(self.net(x))


class GaussianFourierFeatureTransform(nn.Module):
    def __init__(self, input_dim=3, mapping_size=128, scale=1.0):  # 稳健的 1.0
        super().__init__()
        self.register_buffer("B", torch.randn(input_dim, mapping_size) * scale)

    def forward(self, x):
        x_proj = torch.matmul(x, self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class GAMEPINNsDecoupled(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=256):
        super().__init__()
        self.mapping_net = AdaptiveMappingNet(in_dim=in_dim)
        self.fourier_layer = GaussianFourierFeatureTransform(input_dim=in_dim, mapping_size=128, scale=1.0)

        self.adaptive_alpha = nn.Parameter(torch.tensor(1.0))

        self.backbone = nn.Sequential(
            nn.Linear(256, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh()
        )
        self.psi_head = nn.Linear(hidden_dim, 1)
        self.p_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, y, t):
        src = torch.cat((x, y, t), dim=-1)
        xi = self.mapping_net(src)
        feat = self.fourier_layer(xi)

        hidden = self.backbone(feat)
        psi_raw = self.psi_head(hidden)
        p = self.p_head(hidden)
        return psi_raw, p


model = GAMEPINNsDecoupled(in_dim=3, hidden_dim=256).to(device)
optimizer_map = Adam(model.mapping_net.parameters(), lr=1e-4)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameters: {total_params:,} (Stable Cosine-Annealing Setup)")


# ==========================================
# 模块 B: 解算与对称平衡的 Loss
# ==========================================
def get_velocity_with_bounded_adaptive_hard(xt, yt, tt):
    psi_raw, p = model(xt, yt, tt)

    R_cyl = 0.5
    r_sq = xt ** 2 + yt ** 2
    hard_mask = ((r_sq - R_cyl ** 2) / (r_sq + 1.0)) ** 2
    psi = model.adaptive_alpha * hard_mask * psi_raw

    u = torch.autograd.grad(psi, yt, grad_outputs=torch.ones_like(psi), retain_graph=True, create_graph=True)[0]
    v = - torch.autograd.grad(psi, xt, grad_outputs=torch.ones_like(psi), retain_graph=True, create_graph=True)[0]

    return u, v, p, psi


def compute_loss_bounded_causal(xt, yt, tt, ut_target, vt_target, pt_target, use_agm_reg=True):
    u, v, p, psi = get_velocity_with_bounded_adaptive_hard(xt, yt, tt)

    u_t = torch.autograd.grad(u, tt, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_y = torch.autograd.grad(u, yt, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
    u_yy = torch.autograd.grad(u, yt, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]

    v_t = torch.autograd.grad(v, tt, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_x = torch.autograd.grad(v, xt, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_y = torch.autograd.grad(v, yt, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_xx = torch.autograd.grad(v, xt, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
    v_yy = torch.autograd.grad(v, yt, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]

    p_x = torch.autograd.grad(p, xt, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_y = torch.autograd.grad(p, yt, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

    f_u = u_t + (u * u_x + v * u_y) + p_x - 0.01 * (u_xx + u_yy)
    f_v = v_t + (u * v_x + v * v_y) + p_y - 0.01 * (v_xx + v_yy)

    # 恢复对称均等权重（u 和 v 均为 1.0）
    loss_data_u = torch.mean((u - ut_target) ** 2) + torch.mean((v - vt_target) ** 2)
    loss_data_p = torch.mean((p - pt_target) ** 2)

    # 因果路由掩码
    num_bins = 20
    t_flat = tt.detach().flatten()
    t_min, t_max = t_flat.min(), t_flat.max()
    bin_edges = torch.linspace(t_min, t_max, num_bins + 1, device=device)

    pde_residuals = f_u ** 2 + f_v ** 2
    bin_losses = []
    for b in range(num_bins):
        mask_b = (t_flat >= bin_edges[b]) & (t_flat < bin_edges[b + 1])
        if mask_b.sum() > 0:
            bin_losses.append(pde_residuals[mask_b].mean())
        else:
            bin_losses.append(torch.tensor(0.0, device=device))

    causal_weights = []
    cumulative_loss = torch.tensor(0.0, device=device)
    for bl in bin_losses:
        w = torch.exp(-10.0 * cumulative_loss.detach())
        causal_weights.append(w)
        cumulative_loss = cumulative_loss + bl

    loss_pde = 0.0
    for b in range(num_bins):
        mask_b = (t_flat >= bin_edges[b]) & (t_flat < bin_edges[b + 1])
        if mask_b.sum() > 0:
            loss_pde = loss_pde + causal_weights[b] * pde_residuals[mask_b].mean()

    loss = loss_data_u + loss_data_p + loss_pde

    if use_agm_reg:
        src = torch.cat([xt, yt, tt], dim=-1)
        xi = model.mapping_net(src)
        dxi_dx = torch.autograd.grad(xi[:, 0].sum(), xt, create_graph=True)[0]
        dxi_dy = torch.autograd.grad(xi[:, 0].sum(), yt, create_graph=True)[0]
        jac_det = torch.abs(dxi_dx * dxi_dy)
        loss_equi = torch.std(jac_det) / (torch.mean(jac_det) + 1e-8)
        loss_barrier = torch.mean(torch.relu(0.03 - jac_det) ** 2)
        loss = loss + 1e-4 * loss_equi + 10.0 * loss_barrier

    return loss


# ==========================================
# 模块 C: RAR 引擎 (增补 1500 点)
# ==========================================
def adaptive_residual_resampling(num_candidates=30000, num_add=1500):
    print(f"\n>>> 启动 RAR 引擎: 全域扫描并增补 1500 个高残差尾迹点...")
    rand_idx = np.random.choice(N * T, num_candidates, replace=False)
    xc = torch.tensor(x[rand_idx, :], dtype=torch.float32, requires_grad=True).to(device)
    yc = torch.tensor(y[rand_idx, :], dtype=torch.float32, requires_grad=True).to(device)
    tc = torch.tensor(t[rand_idx, :], dtype=torch.float32, requires_grad=True).to(device)

    model.eval()
    uc, vc, pc, _ = get_velocity_with_bounded_adaptive_hard(xc, yc, tc)

    uc_t = torch.autograd.grad(uc, tc, grad_outputs=torch.ones_like(uc), create_graph=True)[0]
    uc_x = torch.autograd.grad(uc, xc, grad_outputs=torch.ones_like(uc), create_graph=True)[0]
    uc_y = torch.autograd.grad(uc, yc, grad_outputs=torch.ones_like(uc), create_graph=True)[0]
    uc_xx = torch.autograd.grad(uc, xc, grad_outputs=torch.ones_like(uc_x), create_graph=True)[0]
    uc_yy = torch.autograd.grad(uc, yc, grad_outputs=torch.ones_like(uc_y), retain_graph=True)[0]
    pc_x = torch.autograd.grad(pc, xc, grad_outputs=torch.ones_like(pc), retain_graph=True)[0]

    f_uc = uc_t + (uc * uc_x + vc * uc_y) + pc_x - 0.01 * (uc_xx + uc_yy)
    err_mag = torch.abs(f_uc).detach().cpu().numpy().flatten()

    top_idx = np.argsort(err_mag)[-num_add:]

    global x_train, y_train, t_train, u_train, v_train, p_train
    x_add = xc[top_idx].detach().requires_grad_(True)
    y_add = yc[top_idx].detach().requires_grad_(True)
    t_add = tc[top_idx].detach().requires_grad_(True)

    u_add = torch.tensor(u[rand_idx[top_idx], :], dtype=torch.float32, requires_grad=True).to(device)
    v_add = torch.tensor(v[rand_idx[top_idx], :], dtype=torch.float32, requires_grad=True).to(device)
    p_add = torch.tensor(p[rand_idx[top_idx], :], dtype=torch.float32, requires_grad=True).to(device)

    x_train = torch.cat([x_train, x_add], dim=0)
    y_train = torch.cat([y_train, y_add], dim=0)
    t_train = torch.cat([t_train, t_add], dim=0)
    u_train = torch.cat([u_train, u_add], dim=0)
    v_train = torch.cat([v_train, v_add], dim=0)
    p_train = torch.cat([p_train, p_add], dim=0)
    print(f"成功增补残差死角样本点: {num_add} 个，当前总训练点数: {x_train.shape[0]}")
    model.train()


# ==========================================
# 模块 D: 两阶段训练 (Phase 2 引入余弦退火调度器)
# ==========================================
print("\n--- [Phase 1] 稳健联合训练 ---")
optimizer_adam = Adam(model.parameters(), lr=1e-3)
for i in tqdm(range(10000), desc="Phase 1 (Stable Adam)"):
    optimizer_adam.zero_grad()
    optimizer_map.zero_grad()
    loss = compute_loss_bounded_causal(x_train, y_train, t_train, u_train, v_train, p_train, use_agm_reg=True)
    loss.backward()
    optimizer_adam.step()
    optimizer_map.step()

adaptive_residual_resampling(num_candidates=30000, num_add=1500)

print("\n--- [Phase 2] 冻结网格，引入余弦退火精细微调 8000 步 ---")
for param in model.mapping_net.parameters():
    param.requires_grad = False

optimizer_adam_p2 = Adam(
    list(model.backbone.parameters()) + list(model.psi_head.parameters()) + list(model.p_head.parameters()), lr=5e-5)
# 配置余弦退火学习率，末端降至 1e-6
scheduler_p2 = CosineAnnealingLR(optimizer_adam_p2, T_max=8000, eta_min=1e-6)

for i in tqdm(range(8000), desc="Phase 2 (Cosine Annealing Fine-tune)"):
    optimizer_adam_p2.zero_grad()
    loss = compute_loss_bounded_causal(x_train, y_train, t_train, u_train, v_train, p_train, use_agm_reg=False)
    loss.backward()
    optimizer_adam_p2.step()
    scheduler_p2.step()

print("\n--- [Phase 3] L-BFGS 极致收敛冲刺 ---")
optimizer_lbfgs = LBFGS(
    list(model.backbone.parameters()) + list(model.psi_head.parameters()) + list(model.p_head.parameters()),
    lr=1.0, max_iter=600, history_size=50, line_search_fn="strong_wolfe"  #max=2000
)
for epoch in range(2000):
    def closure():
        optimizer_lbfgs.zero_grad()
        loss = compute_loss_bounded_causal(x_train, y_train, t_train, u_train, v_train, p_train, use_agm_reg=False)
        loss.backward()
        return loss
    optimizer_lbfgs.step(closure)

# 保存模型
torch.save(model.state_dict(), './ns_game_pinn_stable_cosine.pt')

# ==========================================
# 模块 E: 测试与评估
# ==========================================
snap = np.array([100])
x_star = X_star[:, 0:1]
y_star = X_star[:, 1:2]
t_star = TT[:, snap]

u_star = U_star[:, 0, snap]
v_star = U_star[:, 1, snap]
p_star = P_star[:, snap]

x_star = torch.tensor(x_star, dtype=torch.float32, requires_grad=True).to(device)
y_star = torch.tensor(y_star, dtype=torch.float32, requires_grad=True).to(device)
t_star = torch.tensor(t_star, dtype=torch.float32, requires_grad=True).to(device)

model.eval()
u_pred, v_pred, p_pred, _ = get_velocity_with_bounded_adaptive_hard(x_star, y_star, t_star)

u_pred = u_pred.cpu().detach().numpy()
v_pred = v_pred.cpu().detach().numpy()
p_pred = p_pred.cpu().detach().numpy()

error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)

end_time = time.time()
total_time = end_time - start_time
minutes = int(total_time // 60)
seconds = total_time % 60

print("\n" + "=" * 60)
print(" GAME-PINN (STABLE COSINE-ANNEALING REPORT) ")
print("=" * 60)
print(f" Total Parameters:      {total_params:,}")
print(f" Trainable Parameters:  {trainable_params:,}")
print(f" Adaptive Alpha Value:  {model.adaptive_alpha.item():.4f}")
print("-" * 60)
print(f" Relative L2 Error (u): {error_u:.6e}")
print(f" Relative L2 Error (v): {error_v:.6e}")
print(f" Relative L2 Error (p): {error_p:.6e}")
print("-" * 60)
print(f" Total Computation Time: {minutes} min {seconds:.2f} sec")
print("=" * 60 + "\n")