# -*- coding: utf-8 -*-
"""
File: main_pipeline.py
Description: GAME-PINN 统一消融实验解算主入口 (架构Bug彻底修复版)
"""
import os
os.environ["DDE_BACKEND"] = "pytorch"

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import inspect
import torch
import numpy as np
import deepxde as dde
from scipy.io import loadmat
import matplotlib.pyplot as plt

from models import CombinedNet, AdaptiveMappingNet, GaussianFourierFeatureTransform, Co_AGM_Callback_Universal
from pde_library import get_pde_problem, bind_hard_constraints
# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 数据/输出目录
DATA_DIR = os.path.join(os.path.dirname(script_dir), "dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(script_dir), "output")

dde.config.set_random_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 全局运行配置开关
# ==============================================================================
#CURRENT_EQUATION = "1D_Burgers"
#CURRENT_EQUATION = "2D_Poisson"
CURRENT_EQUATION = "1D_Allen_Cahn"
POISSON_A_PARAM = 10
EPOCHS_PHASE1 = 12000


class LossHistoryTracker(dde.callbacks.Callback):
    def __init__(self, X_test, y_test, period=100):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.period = period

        self.epochs = []
        self.errors = []

    def on_epoch_end(self):
        step = self.model.train_state.iteration
        if step % self.period == 0:
            y_pred = self.model.predict(self.X_test)
            err = dde.metrics.l2_relative_error(self.y_test, y_pred)
            self.epochs.append(step)
            self.errors.append(err)
            print(f"[Tracker] Step {step}: Relative L2 Error = {err:.4e}")

    def save_history(self, filename_prefix="game_pinn_burgers"):
        np.save(f"{filename_prefix}_epochs.npy", np.array(self.epochs))
        np.save(f"{filename_prefix}_errors.npy", np.array(self.errors))
        print(f"数据已成功保存至 {filename_prefix}_epochs.npy 和 {filename_prefix}_errors.npy")


def adaptive_residual_resampling(model, geom_domain, pde_residual, num_candidates=60000, num_add=2500):
    """通用对抗残差点精细重采样引擎 (RAR)"""
    print(f"\n>>> 启动 RAR 引擎: 全域流场扫描寻找误差死角...")
    X_candidates = geom_domain.random_points(num_candidates)
    num_args = len(inspect.signature(pde_residual).parameters)

    def safe_operator(inputs, outputs):
        if num_args >= 3:
            return pde_residual(inputs, outputs, None)
        else:
            return pde_residual(inputs, outputs)

    f_residual = model.predict(X_candidates, operator=safe_operator)
    if isinstance(f_residual, (list, tuple)):
        f_residual = np.column_stack(f_residual)

    err_mag = np.linalg.norm(f_residual, axis=1)
    idx_top = np.argsort(err_mag)[-num_add:]
    X_to_add = X_candidates[idx_top]

    model.data.add_anchors(X_to_add)
    print(f"成功锁定并硬增补物理死角样本点: {num_add} 个")


def main():
    print("=" * 70)
    print(f" 正在直接加载通用物理引擎解算任务: [ {CURRENT_EQUATION} ] ")
    print("=" * 70)

    geom_domain, pde_residual, expected_mode = get_pde_problem(CURRENT_EQUATION, POISSON_A_PARAM)

    in_dim = geom_domain.dim
    out_dim = 1

    m_net = AdaptiveMappingNet(in_dim=in_dim, equation_name=CURRENT_EQUATION).to(device)
    f_layer = GaussianFourierFeatureTransform(input_dim=in_dim, mapping_size=128, scale=5.0).to(device)
    p_net = dde.nn.FNN([256] + [128] * 5 + [out_dim], "swish", "Glorot normal")

    net = CombinedNet(m_net, f_layer, p_net)

    # ==================================================
    # 📌 新增：统计并打印 GAME-PINN 总可训练参数量
    # ==================================================
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"\n[Model Architecture Info] GAME-PINN Total Trainable Parameters: {total_params:,}\n")
    # ==================================================

    # 实施硬锁定，拦截 forward 内的局部微小方差对全局因果策略的无差覆盖
    net.current_routing_mode = expected_mode
    net.force_routing_lock = True

    print(f"DEBUG: 强制路由模式已重置并锁定为: {net.current_routing_mode}")

    bind_hard_constraints(net, CURRENT_EQUATION)
    print(f"DEBUG: 当前绑定的 Ansatz 函数: {net.output_transform.__name__}")

    num_pts = 7000 if CURRENT_EQUATION == "1D_Burgers" else 8000  #7000
    num_tst = 2000
    data = dde.data.PDE(geom_domain, pde_residual, [], num_domain=num_pts, num_test=num_tst)

    def pure_pde_losses_bypass(targets, outputs, loss_fn, inputs, model, aux=None):
        num_args = len(inspect.signature(pde_residual).parameters)
        f = pde_residual(inputs, outputs, aux) if num_args >= 3 else pde_residual(inputs, outputs)
        if not isinstance(f, (list, tuple)):
            f = [f]
        return [loss_fn(torch.zeros_like(fi), fi) for fi in f]

    data.losses = pure_pde_losses_bypass
    data.losses_train = pure_pde_losses_bypass

    model = dde.Model(data, net)
    wall_clock_start = time.time()


    # ==================================================
    # Phase 1: 协同演化训练阶段 (Adam + AGM)
    # ==================================================
    print(f"\n>>> Phase 1: 统一多维 AGM 网格演化与主干网联合训练 {EPOCHS_PHASE1} steps")
    model.compile("adam", lr=1e-3)

    agm_callback = Co_AGM_Callback_Universal(m_net, log_every=100, equation_name=CURRENT_EQUATION)

    t0 = time.time()
    model.train(iterations=EPOCHS_PHASE1, callbacks=[agm_callback])
    time_p1 = time.time() - t0
    adaptive_residual_resampling(model, geom_domain, pde_residual, num_candidates=60000, num_add=2500)

    # ==================================================
    # Phase 2: 冻结网格，微调物理层 (Fine-tune)
    # ==================================================
    print("\n>>> Phase 2: 冻结流形网格层，集中攻坚物理高梯度区残差")
    for param in m_net.parameters():
        param.requires_grad = False

    model.compile("adam", lr=1e-4)
    t1 = time.time()
    model.train(iterations=5000)
    time_p2 = time.time() - t1

    # ==================================================
    # Phase 3: L-BFGS 极致收敛冲刺
    # ==================================================
    print("\n>>> Phase 3: L-BFGS 变分极小化梯度级优化...")
    t2 = time.time()

    dde.optimizers.config.set_LBFGS_options(
        maxcor=50,
        maxiter=3000,
        ftol=1e-13,
        gtol=1e-10
    )
    model.compile("L-BFGS")
    losshistory, train_state = model.train()
    time_p3 = time.time() - t2
    wall_clock_end = time.time()

    # ==================================================
    # Phase 4: 全域后处理与 L2 相对误差评估
    # ==================================================
    print("\n>>> Phase 4: 评估全局 L2 相对误差...")
    error_str = "N/A"

    try:
        if CURRENT_EQUATION == "1D_Burgers":
            burgers_path = os.path.join(DATA_DIR, "Burgers.npz")
            if os.path.exists(burgers_path):
                curr_data = np.load(burgers_path)
                t_flat, x_flat = curr_data["t"].flatten(), curr_data["x"].flatten()
                u_sol = curr_data["usol"].T

                T_grid, X_grid = np.meshgrid(t_flat, x_flat, indexing='ij')
                X_test = np.hstack((X_grid.reshape(-1, 1), T_grid.reshape(-1, 1))).astype(np.float32)
                y_test = u_sol.reshape(-1, 1).astype(np.float32)

                u_pred = model.predict(X_test)
                l2_err = dde.metrics.l2_relative_error(y_test, u_pred)
                error_str = f"{l2_err:.4e}"
                print(f"    [Burgers 测试成功] 测试节点数: {X_test.shape[0]}")

                plt.rcParams['font.family'] = 'serif'
                plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
                plt.rcParams['axes.unicode_minus'] = False

                u_pred_grid = u_pred.reshape(len(t_flat), len(x_flat))
                error = np.abs(u_sol - u_pred_grid)

                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.pcolormesh(T_grid, X_grid, u_pred_grid, cmap='jet', shading='gouraud')
                plt.colorbar(label='Predicted u')
                plt.title("AGM-PINN + Spatial Hardness Prediction")
                plt.xlabel("t")
                plt.ylabel("x")

                plt.subplot(1, 2, 2)
                plt.pcolormesh(T_grid, X_grid, error, cmap='Reds', shading='gouraud')
                plt.colorbar(label='Abs Error')
                plt.title("Error Distribution (Adaptive Hardness)")
                plt.xlabel("t")
                plt.ylabel("x")

                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, "error_distribution_spatial_hardness.png"))
                plt.close()
            else:
                print("    [警告] 找不到 Burgers.npz，跳过评估。")


        elif CURRENT_EQUATION == "2D_Poisson":
            def gen_test_x(num):
                x = np.linspace(0, 1, num)
                y = np.linspace(0, 1, num)
                xv, yv = np.meshgrid(x, y)
                return np.stack([xv.flatten(), yv.flatten()], axis=1).astype(np.float32)
            def sol(t):
                x, y = t[:, 0:1], t[:, 1:2]
                return (16 * x * y * (1 - x) * (1 - y)) ** POISSON_A_PARAM
            num_test = 100
            X_test = gen_test_x(num_test)
            y_true = sol(X_test)
            u_pred = model.predict(X_test)
            l2_err = dde.metrics.l2_relative_error(y_true, u_pred)
            error_str = f"{l2_err:.4e}"
            print(f"    [Poisson 测试成功] 测试节点数: {X_test.shape[0]}")
            print(f"    [Poisson 相对L2误差]: {error_str}")
            # 绘图可视化
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Times New Roman']
            u_true_grid = y_true.reshape(num_test, num_test)
            u_pred_grid = u_pred.reshape(num_test, num_test)
            error_grid = np.abs(u_true_grid - u_pred_grid)
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            # 绘制真实解
            c1 = axes[0].imshow(u_true_grid, extent=[0, 1, 0, 1], origin='lower', cmap='jet')
            axes[0].set_title("True Solution")
            plt.colorbar(c1, ax=axes[0])
            # 绘制预测解
            c2 = axes[1].imshow(u_pred_grid, extent=[0, 1, 0, 1], origin='lower', cmap='jet')
            axes[1].set_title("Predicted Solution (GAME-PINN)")
            plt.colorbar(c2, ax=axes[1])
            # 绘制误差分布
            c3 = axes[2].imshow(error_grid, extent=[0, 1, 0, 1], origin='lower', cmap='Reds')
            axes[2].set_title(f"Absolute Error (L2: {error_str})")
            plt.colorbar(c3, ax=axes[2])
            plt.tight_layout()
            plt.savefig("poisson_visualization.png")
            plt.close()
            print("    [可视化] 绘图已保存至 poisson_visualization.png")




        elif CURRENT_EQUATION == "1D_Allen_Cahn":
            mat_path = os.path.join(DATA_DIR, "usol_D_0.001_k_5.mat")
            if os.path.exists(mat_path):
                data_mat = loadmat(mat_path)
                u_raw = data_mat["u"]
                x_flat = data_mat["x"].flatten()
                t_flat = data_mat["t"].flatten()
                nx, nt = len(x_flat), len(t_flat)

                u_true = u_raw if u_raw.shape == (nx, nt) else u_raw.T
                X_grid, T_grid = np.meshgrid(x_flat, t_flat, indexing='xy')
                X_test = np.vstack((X_grid.flatten(), T_grid.flatten())).T.astype(np.float32)
                y_test = u_true.T.reshape(-1, 1).astype(np.float32)

                net.eval()
                with torch.no_grad():
                    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                    u_pred_flat_tensor = net(X_test_tensor)
                    u_pred_flat = u_pred_flat_tensor.cpu().numpy()

                l2_err = dde.metrics.l2_relative_error(y_test, u_pred_flat)
                error_str = f"{l2_err:.4e}"
                print(f"    [Allen-Cahn 测试成功] 严格对齐测试节点数: {X_test.shape[0]}")

                u_pred = u_pred_flat.reshape(nt, nx).T
                T_mesh, X_mesh = np.meshgrid(t_flat, x_flat)
                error_map = np.abs(u_true - u_pred)

                plt.figure(figsize=(13, 5))
                plt.subplot(1, 2, 1)
                plt.pcolormesh(T_mesh, X_mesh, u_pred, cmap='jet', shading='gouraud')
                plt.colorbar(label='Predicted u(x,t)')
                plt.title("GAME-PINN Unified Prediction (Allen-Cahn)")
                plt.xlabel("t")
                plt.ylabel("x")

                plt.subplot(1, 2, 2)
                plt.pcolormesh(T_mesh, X_mesh, error_map, cmap='Reds', shading='gouraud')
                plt.colorbar(label='Absolute Error')
                plt.title(f"True L2 Error = {l2_err:.2e}")
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, "allencahn_unified_error.png"))
                plt.close()
            else:
                print(f"    [警告] 找不到参考解矩阵 {mat_path}，跳过评估。")

    except Exception as e:
        print(f"    [计算失败] 误差计算崩溃: {str(e)}")
        error_str = "计算失败"

    print("\n" + "=" * 60)
    print("      ACADEMIC PERFORMANCE REPORT (UNIFIED FRAMEWORK)")
    print("=" * 60)
    print(f"时域触发器状态识别 : {net.current_routing_mode} (预期: {expected_mode})")
    print(f"总计物理优化步数   : {train_state.step} steps")
    print(f"Phase 1 耗时 (Adam) : {time_p1:10.2f} s")
    print(f"Phase 2 耗时 (Fine) : {time_p2:10.2f} s")
    print(f"Phase 3 耗时 (BFGS) : {time_p3:10.2f} s")
    print(f"端到端全量时钟总耗时: {wall_clock_end - wall_clock_start:10.2f} s")
    print(f"最终相对 L2 误差 : {error_str}")
    print("=" * 60)


if __name__ == "__main__":
    main()