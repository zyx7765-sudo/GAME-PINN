# -*- coding: utf-8 -*-
"""
File: pde_library.py
Description: 标准化多物理场PDE方程及硬约束控制边界库（集成时域显式因果门控Ansatz版）
"""
import torch
import numpy as np
import deepxde as dde

nu = 0.01 / np.pi  # Burgers 方程粘性系数


def get_pde_problem(equation_name, a_param=10):
    """
    一键返回对应的计算几何域、PDE残差定义算子和预期模式
    """
    if equation_name == "1D_Burgers":
        geom = dde.geometry.Interval(-1, 1)
        timedomain = dde.geometry.TimeDomain(0, 1)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)

        def pde_func(x, y, net):
            u_x = dde.grad.jacobian(y, x, i=0, j=0)
            u_t = dde.grad.jacobian(y, x, i=0, j=1)
            u_xx = dde.grad.hessian(y, x, i=0, j=0)

            res_raw = u_t + y * u_x - nu * u_xx

            if hasattr(net, "current_routing_mode") and net.current_routing_mode == "CAUSAL":
                routing_w = net.get_routing_and_causal_weights(x)
                mod_factor = 1.0 + 0.5 * (routing_w - 1.0)
                return res_raw * mod_factor
            return res_raw

        return geomtime, pde_func
        
    elif equation_name == "2D_Poisson":
        geom = dde.geometry.Rectangle([0, 0], [1, 1])

        def pde_func(x, y, net):
            du_xx = dde.grad.hessian(y, x, i=0, j=0)
            du_yy = dde.grad.hessian(y, x, i=1, j=1)

            x_coor, y_coor = x[:, 0:1], x[:, 1:2]
            term = (x_coor * (1.0 - x_coor) * y_coor * (1.0 - y_coor))
            eps = 1e-12

            u_xx = (16 ** a_param * a_param * (
                        a_param * (1.0 - 2.0 * x_coor) ** 2 - 2.0 * x_coor ** 2 + 2.0 * x_coor - 1.0) * (
                                term ** a_param) / ((x_coor - 1.0) ** 2 * x_coor ** 2 + eps))
            u_yy = (16 ** a_param * a_param * (
                        a_param * (1.0 - 2.0 * y_coor) ** 2 - 2.0 * y_coor ** 2 + 2.0 * y_coor - 1.0) * (
                                term ** a_param) / ((y_coor - 1.0) ** 2 * y_coor ** 2 + eps))

            f_src = - (u_xx + u_yy)
            return du_xx + du_yy + f_src

        return geom, pde_func

    elif equation_name == "1D_Allen_Cahn":
        geom = dde.geometry.Interval(-1, 1)
        timedomain = dde.geometry.TimeDomain(0, 1)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)

        def pde_func(x, y, net):
            u_t = dde.grad.jacobian(y, x, i=0, j=1)
            u_xx = dde.grad.hessian(y, x, i=0, j=0)
            res_raw = u_t - 0.001 * u_xx + 5.0 * (y ** 3 - y)

            if hasattr(net, "current_routing_mode") and net.current_routing_mode == "CAUSAL":
                routing_w = net.get_routing_and_causal_weights(x)
                mod_factor = 1.0 + 0.5 * (routing_w - 1.0)
                return res_raw * mod_factor
            return res_raw

        return geomtime, pde_func

    else:
        raise ValueError(f"未知的物理方程类型: {equation_name}")


# ==================================================
# 严格物理边界硬约束 Ansatz (Output Transforms)
# ==================================================
def burgers_hard_constraint(x, y_pred, xi, combined_net_ref):
    """Burgers 方程专用：时空边界动态平滑硬约束"""
    x_raw, t_raw = x[:, 0:1], x[:, 1:2]
    g_x = -torch.sin(np.pi * x_raw)

    time_signal = torch.tanh(2.0 * t_raw)
    spatial_signal = 1.0 - x_raw ** 2
    gate_input = torch.cat([time_signal, spatial_signal], dim=1)

    k_raw = combined_net_ref.hardness_gate(gate_input)
    k_time_adaptive = 5.0 + 40.0 * torch.sigmoid(k_raw)

    dist_x = 1.0 - x_raw ** 2
    dist_t = t_raw

    ansatz = g_x + dist_t * y_pred * (1.0 - torch.exp(-k_time_adaptive * dist_x))
    return ansatz


def poisson_hard_constraint(x, y_pred, xi, combined_net_ref):
    """2D 泊松方程专用：空间边界 100% 锁死 Ansatz"""
    x_raw, y_raw = x[:, 0:1], x[:, 1:2]
    return x_raw * y_raw * (1.0 - x_raw) * (1.0 - y_raw) * y_pred


def allencahn_hard_constraint(x, y_pred, xi, combined_net_ref):
    """1D Allen-Cahn 方程专用：自适应时空带宽平滑硬约束"""
    x_raw, t_raw = x[:, 0:1], x[:, 1:2]
    g_x = (x_raw ** 2) * torch.cos(np.pi * x_raw)

    local_res = combined_net_ref.get_residual_indicator(x_raw, t_raw)
    kappa_input = torch.cat([x_raw, t_raw, local_res], dim=1)
    kappa_raw = combined_net_ref.kappa_net(kappa_input)
    kappa = 5.0 + 40.0 * torch.sigmoid(kappa_raw)

    dist_x = 1.0 - torch.exp(-kappa * (1.0 - x_raw ** 2))
    dist_t = torch.tanh(30.0 * (t_raw ** 2))
    return g_x + dist_x * dist_t * y_pred


def bind_hard_constraints(net, equation_name):
    """根据物理方程自动绑定硬约束转换器"""
    if equation_name == "1D_Burgers":
        net.apply_output_transform(burgers_hard_constraint)
    elif equation_name == "2D_Poisson":
        net.apply_output_transform(poisson_hard_constraint)
    elif equation_name == "1D_Allen_Cahn":
        net.apply_output_transform(allencahn_hard_constraint)
