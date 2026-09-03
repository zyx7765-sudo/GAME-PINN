# -*- coding: utf-8 -*-
"""
File: models.py
Description: 统一神经网络架构底座
             
"""
import torch
import torch.nn as nn
import numpy as np
import deepxde as dde
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdaptiveMappingNet(nn.Module):
    """
    通用多维自适应流形映射网络 (AGM)
    将物理空间坐标 x 映射到计算域流形 xi，通过残差控制其空间伸缩
    """

    def __init__(self, in_dim=2, equation_name="2D_Poisson"):
        super().__init__()
        self.equation_name = equation_name

        # 针对 1D 时空演化方程，空间网格变形网络的第一层输入/输出维度解耦为空间物理维度 1
        is_spatiotemporal = equation_name in ["1D_Allen_Cahn", "1D_Burgers"]
        current_in_dim = 1 if is_spatiotemporal else in_dim
        out_dim = 1 if is_spatiotemporal else in_dim

        # 搭建一个多层感知机（MLP）：输入层→64维→双曲正切激活→64维隐藏层→双曲正切激活→输出层
        # 双曲正切输出范围是(-1, 1)，适合生成微小的位移量
        self.net = nn.Sequential(
            nn.Linear(current_in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, out_dim)
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        # 如果是时空演化方程，只对空间坐标进行流形自适应映射，时间轴保持刚性正交推进
        if self.equation_name in ["1D_Allen_Cahn", "1D_Burgers"]:
            x_raw, t_raw = x[:, 0:1], x[:, 1:2]
            dx = 0.2 * torch.tanh(self.net(x_raw))
            return torch.cat([x_raw + dx, t_raw], dim=1)
        return x + self.net(x)


class GaussianFourierFeatureTransform(nn.Module):
    """
    可学习/固定高斯傅里叶特征嵌入层 (FFM)
    解决 PINN 对高频强定域尖峰及剧烈激波前沿的“光谱偏差”恶性收敛问题
    """

    def __init__(self, input_dim=2, mapping_size=128, scale=5.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.register_buffer("B", torch.randn(input_dim, mapping_size) * scale)

    def forward(self, x):
        x_proj = torch.matmul(x, self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class CombinedNet(nn.Module):
    """
    【算子本征物理模态路由与拓扑因果门控机制 (EOTR)】
    """
    regularizer = None

    def __init__(self, mapping_net, fourier_layer, physics_net, max_points=15000):
        super().__init__()
        self.mapping_net = mapping_net
        self.fourier_layer = fourier_layer
        self.physics_net = physics_net
        self.output_transform = None

        self.hardness_gate = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

        self.kappa_net = nn.Sequential(
            nn.Linear(3, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )

        self.register_buffer('tau', torch.tensor(0.0))
        self.register_buffer('buffer_coords', torch.zeros(max_points, 2))
        self.register_buffer('buffer_S_indicator', torch.zeros(max_points, 1))
        self.current_buffer_size = 0
        self.current_routing_mode = "BYPASS"   # 默认为稳态
        self.force_routing_lock = False         # 初始为 False，以便第一次前向时允许路由判定

    def apply_output_transform(self, transform):
        self.output_transform = transform

    def update_residual_buffer(self, coords, S_values):
        n_points = coords.shape[0]
        if n_points > self.buffer_coords.shape[0]:
            self.buffer_coords = torch.zeros(n_points, 2, device=coords.device)
            self.buffer_S_indicator = torch.zeros(n_points, 1, device=coords.device)

        self.buffer_coords[:n_points].copy_(coords.detach())
        safe_S = torch.where(torch.isnan(S_values), torch.zeros_like(S_values), S_values)
        self.buffer_S_indicator[:n_points].copy_(safe_S.detach())
        self.current_buffer_size = n_points

    def get_residual_indicator(self, x, t):
        if self.current_buffer_size == 0:
            return torch.zeros_like(x)
        coords_query = torch.cat([x, t], dim=1)
        coords_train = self.buffer_coords[:self.current_buffer_size]
        dists = torch.cdist(coords_query, coords_train)
        min_idx = torch.argmin(dists, dim=1)
        return self.buffer_S_indicator[min_idx]

    def forward(self, x):
        # ====== 【路由判定逻辑】 ======
        if not self.force_routing_lock:
            # 如果输入维度小于 2（纯空间问题），或者时间维度没有变化（全是 t=0 时刻），则走 BYPASS
            if x.shape[1] < 2:
                self.current_routing_mode = "BYPASS"
            else:
                # 检测时间维的数值变化范围
                t_col = x[:, -1:]
                t_var = torch.var(t_col)
                if t_var < 1e-7:  # 如果时间轴没有变化，判定为稳态快照
                    self.current_routing_mode = "BYPASS"
                else:
                    self.current_routing_mode = "CAUSAL"

        # 流形映射 → 傅里叶特征 → 物理网络
        xi = self.mapping_net(x)
        feat = self.fourier_layer(xi)
        res = self.physics_net(feat)

        if self.output_transform is not None:
            return self.output_transform(x, res, xi, self)
        return res

    # ================================================================
    # 算法自动算子拓扑探测
    # ================================================================
    def auto_detect_routing_mode(self, pde_func, x_sample_np, time_dim_index=None):
        """
        基于自动微分，自动计算 PDE 算子对时间导数项 u_t 的 Frobenius 范数敏感度。
        参数:
            pde_func: 来自 pde_library 的残差函数，签名为 pde_func(x, y, net)
            x_sample_np: 用于探测的样本点 (numpy数组 或 Tensor)
            time_dim_index: 时间维索引，默认取最后一维
        返回:
            str: "CAUSAL" 或 "BYPASS"
        """
        if not isinstance(x_sample_np, torch.Tensor):
            x_sample = torch.tensor(x_sample_np, dtype=torch.float32, device=next(self.parameters()).device)
        else:
            x_sample = x_sample_np.to(next(self.parameters()).device)
        
        x_sample.requires_grad_(True)
        original_lock_state = self.force_routing_lock
        self.force_routing_lock = False  # 让 forward 自由探测
        u = self(x_sample)  # shape: [N, 1]

        self.force_routing_lock = original_lock_state
 
        u_x = torch.autograd.grad(
            outputs=u,
            inputs=x_sample,
            grad_outputs=torch.ones_like(u),
            create_graph=True,  
            retain_graph=True
        )[0]  # shape: [N, dim]

        if time_dim_index is None:
            time_dim_index = x_sample.shape[1] - 1
        u_t = u_x[:, time_dim_index:time_dim_index+1]  # shape: [N, 1]
        R_raw = pde_func(x_sample, u, self)
        
        if isinstance(R_raw, (list, tuple)):
            # 将所有残差项横向拼接，然后求整体 MSE
            R_cat = torch.cat([r.reshape(-1, 1) for r in R_raw], dim=1)
            loss_res = torch.mean(R_cat ** 2)
        else:
            R = R_raw.reshape(-1, 1)
            loss_res = torch.mean(R ** 2)

        grad_wrt_ut = torch.autograd.grad(
            outputs=loss_res,
            inputs=u_t,
            retain_graph=False,    # 探测结束，立即释放计算图以节省显存
            allow_unused=True      # 如果 u_t 未被使用，返回 None 而不报错
        )[0]

        if grad_wrt_ut is None:
            sensitivity_norm = 0.0
        else:
            sensitivity_norm = torch.norm(grad_wrt_ut).item()

        threshold = 1e-6
        if sensitivity_norm < threshold:
            detected_mode = "BYPASS"
        else:
            detected_mode = "CAUSAL"

        self.current_routing_mode = detected_mode
        self.force_routing_lock = True

        print(f"[Auto-Router] 算子敏感度范数: {sensitivity_norm:.2e} | 判定模式: {detected_mode}")
        return detected_mode

    def get_routing_and_causal_weights(self, x_raw):
        """
        基于局域时空演化度量 (LSEM) 的连续因果门控函数
        """
        if self.current_routing_mode == "BYPASS":
            return torch.ones((x_raw.shape[0], 1), device=x_raw.device)

        t = x_raw[:, -1:]
        spatial_coords = x_raw[:, :-1]
        lsem = self.get_residual_indicator(spatial_coords, t)
        # 公式：w = 1.0 + exp(-alpha * (LSEM - threshold)) 或基于累计残差的包络线平滑映射
        alpha_coeff = 5.0
        weights = 1.0 + torch.tanh(alpha_coeff * lsem) * t

        weights = weights / (weights.mean() + 1e-8)
        return weights


class Co_AGM_Callback_Universal(dde.callbacks.Callback):
    """
    多维泛化几何等分布正则化演化回调函数
    """

    def __init__(self, mapping_net, log_every=100, equation_name="2D_Poisson"):
        super().__init__()
        self.mapping_net = mapping_net
        self.log_every = log_every
        self.equation_name = equation_name
        self.optimizer_map = torch.optim.Adam(self.mapping_net.parameters(), lr=1e-4)
        self.j_min_safe = 0.03
        self.alpha_barrier = 25.0
        self.prev_kappa = None
        self.kappa_reg_weight = 1e-5

    def on_epoch_end(self):
        if self.model.train_state.iteration % self.log_every != 0:
            return

        x_raw = torch.tensor(self.model.data.train_x, dtype=torch.float32, device=device, requires_grad=True)
        y_pred = self.model.net(x_raw)

        if self.equation_name == "1D_Allen_Cahn":
            u_x = torch.autograd.grad(y_pred.sum(), x_raw, create_graph=True)[0][:, 0:1]
            u_t = torch.autograd.grad(y_pred.sum(), x_raw, create_graph=True)[0][:, 1:2]
            u_xx = torch.autograd.grad(u_x.sum(), x_raw, create_graph=True)[0][:, 0:1]
            grad_mag = torch.abs(u_x).detach()

            f_res = torch.abs(u_t - 0.001 * u_xx + 5.0 * (y_pred ** 3 - y_pred)).detach()
            S_indicator = grad_mag + f_res
            s_max = S_indicator.max()
            S_indicator = S_indicator / s_max if s_max > 1e-6 else torch.zeros_like(S_indicator)
            self.model.net.update_residual_buffer(x_raw[:, 0:2], S_indicator)

        elif self.equation_name == "1D_Burgers":
            u_x = torch.autograd.grad(y_pred.sum(), x_raw, create_graph=True)[0][:, 0:1]
            grad_mag = torch.abs(u_x).detach()
            self.model.net.update_residual_buffer(x_raw[:, 0:2], grad_mag)

        self.optimizer_map.zero_grad()
        xi = self.mapping_net(x_raw)

        dy_dx = torch.autograd.grad(y_pred.sum(), x_raw, create_graph=True)[0]
        grad_mag_uni = torch.norm(dy_dx, dim=1, keepdim=True).detach()
        omega = 1.0 + 4.0 * (grad_mag_uni / (grad_mag_uni.max() + 1e-8))

        # 动态感知输入的实际空间物理维度，切勿将时间轴强制卷入雅可比矩阵
        spatial_dim = x_raw.shape[1] - 1 if self.equation_name in ["1D_Allen_Cahn", "1D_Burgers"] else x_raw.shape[1]

        if spatial_dim == 1:
            # 1D 空间一维演化雅可比
            dxi_dxraw = torch.autograd.grad(xi[:, 0].sum(), x_raw, create_graph=True)[0][:, 0:1]
            jacobian_det = torch.abs(dxi_dxraw)
        else:
            # 2D 静态空间物理雅可比
            dxi_dxraw = torch.autograd.grad(xi[:, 0].sum(), x_raw, create_graph=True)[0]
            deta_dxraw = torch.autograd.grad(xi[:, 1].sum(), x_raw, create_graph=True)[0]
            dxi_dx, dxi_dy = dxi_dxraw[:, 0:1], dxi_dxraw[:, 1:2]
            deta_dx, deta_dy = deta_dxraw[:, 0:1], deta_dxraw[:, 1:2]
            jacobian_det = torch.abs(dxi_dx * deta_dy - dxi_dy * deta_dx)

        weighted_volume = omega * jacobian_det
        loss_equi = torch.std(weighted_volume) / (torch.mean(weighted_volume) + 1e-8)
        loss_relu_guard = torch.mean(torch.relu(self.j_min_safe - jacobian_det) ** 2)

        total_map_loss = 2.0 * loss_equi + self.alpha_barrier * loss_relu_guard

        if self.equation_name == "1D_Allen_Cahn":
            with torch.no_grad():
                local_res = self.model.net.get_residual_indicator(x_raw[:, 0:1], x_raw[:, 1:2])
                kappa_input = torch.cat([x_raw[:, 0:1], x_raw[:, 1:2], local_res], dim=1)
                kappa_current = self.model.net.kappa_net(kappa_input)
            if self.prev_kappa is not None:
                kappa_diff = torch.mean((kappa_current - self.prev_kappa) ** 2)
                total_map_loss = total_map_loss + self.kappa_reg_weight * kappa_diff
            self.prev_kappa = kappa_current.detach()

        if not torch.isnan(total_map_loss):
            total_map_loss.backward()
            self.optimizer_map.step()

        del dy_dx, grad_mag_uni, xi, x_raw, y_pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
