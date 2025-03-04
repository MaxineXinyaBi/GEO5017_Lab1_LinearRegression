import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import argparse


# -------------------------
#  Improvement 1:
#  Decaying Learning Rate
# -------------------------

def poly2(t, a, b, c):
    return a + b * t + c * (t ** 2)

def sse_loss(t_data, x_data, a, b, c):
    errors = x_data - poly2(t_data, a, b, c)
    return 0.5 * np.sum(errors ** 2)

def grad_sse(t_data, x_data, a, b, c):
    errors = x_data - poly2(t_data, a, b, c)
    d_a = -np.sum(errors)
    d_b = -np.sum(errors * t_data)
    d_c = -np.sum(errors * (t_data ** 2))
    return d_a, d_b, d_c

def gradient_descent_poly2(t_data, x_data, lr, epochs, tolerance, a, b, c):
    a_cache, b_cache, c_cache = a, b, c
    a_prev, b_prev, c_prev = 0, 0, 0

    loss_list = []
    epoch_list = []

    epoch_val = 0
    count = 0
    prev_loss = float('inf')

    while epoch_val < epochs:
        count += 1
        a_prev, b_prev, c_prev = a_cache, b_cache, c_cache

        d_a, d_b, d_c = grad_sse(t_data, x_data, a_cache, b_cache, c_cache)

        step_a = lr * d_a
        step_b = lr * d_b
        step_c = lr * d_c
        step_size = np.sqrt(step_a ** 2 + step_b ** 2 + step_c ** 2)

        a_new = a_cache - step_a
        b_new = b_cache - step_b
        c_new = c_cache - step_c

        current_loss = sse_loss(t_data, x_data, a_new, b_new, c_new)

        if current_loss < prev_loss:
            a_cache, b_cache, c_cache = a_new, b_new, c_new
            prev_loss = current_loss
            epoch_val += 1
            loss_list.append(current_loss)
            epoch_list.append(epoch_val)

            if step_size < tolerance:
                break
        else:
            lr = lr / 2
            print("Changing epoch:", count)
            print("New lr:", lr)
            a_cache, b_cache, c_cache = a_prev, b_prev, c_prev
    print("Final iteration:", count)
    return a_cache, b_cache, c_cache, epoch_list, loss_list

# -----------------------------


def main_Decaying_Learning_Rate():
    file_path = "metadata.csv"
    df = pd.read_csv(file_path)
    x_data = np.array(df["x"], dtype=float)
    y_data = np.array(df["y"], dtype=float)
    z_data = np.array(df["z"], dtype=float)

    t_data = np.arange(1, len(x_data) + 1, dtype=float)

    init_lr = 0.01  # Initial learning rate (tunable)
    max_epochs = 60000  # Max allowed iterations (tunable)
    tolerance = 1e-6  # Min allowed step size (tunable)
    a, b, c = 0, 0, 0  # Initial polynomial parameters (tunable)

    print("\nFitting x(t):")
    a_x, b_x, c_x, epochs_x, loss_x = gradient_descent_poly2(t_data, x_data, init_lr, max_epochs, tolerance, a, b, c)
    sse_x = sse_loss(t_data, x_data, a_x, b_x, c_x)

    print("\nFitting y(t):")
    a_y, b_y, c_y, epochs_y, loss_y = gradient_descent_poly2(t_data, y_data, init_lr, max_epochs, tolerance, a, b, c)
    sse_y = sse_loss(t_data, y_data, a_y, b_y, c_y)

    print("\nFitting z(t):")
    a_z, b_z, c_z, epochs_z, loss_z = gradient_descent_poly2(t_data, z_data, init_lr, max_epochs, tolerance, a, b, c)
    sse_z = sse_loss(t_data, z_data, a_z, b_z, c_z)

    # NEW: 计算总SSE
    total_SSE = sse_x + sse_y + sse_z

    print("\nFitted function relationships:")
    print(f"x(t) = {a_x:.6f} + {b_x:.6f} * t + {c_x:.6f} * t^2, SSE_x = {sse_x:.6f}")
    print(f"y(t) = {a_y:.6f} + {b_y:.6f} * t + {c_y:.6f} * t^2, SSE_y = {sse_y:.6f}")
    print(f"z(t) = {a_z:.6f} + {b_z:.6f} * t + {c_z:.6f} * t^2, SSE_z = {sse_z:.6f}")
    print(f"Total SSE = {total_SSE:.6f}")  # NEW: 打印总SSE

    next_t = len(t_data) + 1
    pred_x = poly2(next_t, a_x, b_x, c_x)
    pred_y = poly2(next_t, a_y, b_y, c_y)
    pred_z = poly2(next_t, a_z, b_z, c_z)
    print(f"\nPrediction for t = {next_t}:")
    print(f"Predicted position: x = {pred_x:.6f}, y = {pred_y:.6f}, z = {pred_z:.6f}")

    # -------------------------
    #  3D Plot
    # -------------------------
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(x_data, y_data, z_data, c='red', marker='o', label='Drone location')
    ax.plot(x_data, y_data, z_data, linestyle='--', color='black', label='Flight path')

    t_markers = np.arange(1, next_t + 1)
    x_markers = poly2(t_markers, a_x, b_x, c_x)
    y_markers = poly2(t_markers, a_y, b_y, c_y)
    z_markers = poly2(t_markers, a_z, b_z, c_z)
    ax.scatter(x_markers, y_markers, z_markers, c='green', marker='o', label='Estimated location')

    t_vals = np.linspace(1, next_t, 50)
    x_vals = poly2(t_vals, a_x, b_x, c_x)
    y_vals = poly2(t_vals, a_y, b_y, c_y)
    z_vals = poly2(t_vals, a_z, b_z, c_z)
    ax.plot(x_vals, y_vals, z_vals, c='green', label='Constant acceleration model')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_title('Constant Acceleration Model', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(fontsize=6)
    ax.view_init(azim=30)

    plt.tight_layout()
    plt.show()

# -------------------------
#  Improvement 2:
#  Covariance optimization in three directions
#  +
#  Considering the angular velocity of constant acceleration
# -------------------------


def covariance_penalty(x_pred, y_pred, z_pred, cov_target):
    data = torch.stack([x_pred, y_pred, z_pred], dim=0)
    cov_est = torch.cov(data)
    penalty = torch.sum((cov_est - cov_target) ** 2)
    return penalty


def angular_velocity_penalty_3d(t, params, omega_target=0.0, dt=1.0):
    vx = params['b_x'] + 2 * params['c_x'] * t
    vy = params['b_y'] + 2 * params['c_y'] * t
    vz = params['b_z'] + 2 * params['c_z'] * t
    v = torch.stack([vx, vy, vz], dim=1)
    norm = torch.norm(v, dim=1, keepdim=True) + 1e-6
    v_norm = v / norm
    dot_products = torch.sum(v_norm[1:] * v_norm[:-1], dim=1)
    dot_products = torch.clamp(dot_products, -1.0, 1.0)
    angles = torch.acos(dot_products)
    angular_velocity = angles / dt
    penalty = torch.sum((angular_velocity - omega_target) ** 2)
    return penalty

# -----------------------------


def main_cov_angle_reg():
    file_path = "metadata.csv"
    df = pd.read_csv(file_path)
    x_data_np = np.array(df["x"], dtype=float)
    y_data_np = np.array(df["y"], dtype=float)
    z_data_np = np.array(df["z"], dtype=float)
    t_data_np = np.arange(1, len(x_data_np) + 1, dtype=float)

    # Convert to torch tensor
    t_data = torch.tensor(t_data_np, dtype=torch.float32)
    x_data = torch.tensor(x_data_np, dtype=torch.float32)
    y_data = torch.tensor(y_data_np, dtype=torch.float32)
    z_data = torch.tensor(z_data_np, dtype=torch.float32)

    params = {
        'a_x': torch.tensor(0.0, requires_grad=True),
        'b_x': torch.tensor(0.0, requires_grad=True),
        'c_x': torch.tensor(0.0, requires_grad=True),
        'a_y': torch.tensor(0.0, requires_grad=True),
        'b_y': torch.tensor(0.0, requires_grad=True),
        'c_y': torch.tensor(0.0, requires_grad=True),
        'a_z': torch.tensor(0.0, requires_grad=True),
        'b_z': torch.tensor(0.0, requires_grad=True),
        'c_z': torch.tensor(0.0, requires_grad=True)
    }

    cov_target = torch.tensor([
        [np.var(x_data_np), 0, 0],
        [0, np.var(y_data_np), 0],
        [0, 0, np.var(z_data_np)]
    ], dtype=torch.float32)

    optimizer = optim.Adam(params.values(), lr=0.01)
    num_epochs = 6000

    lambda_cov = 0.0
    lambda_ang = 0.0

    target_ratio_cov = 0.1
    target_ratio_ang = 0.1

    reg_lr = 0.0001

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        x_pred = poly2(t_data, params['a_x'], params['b_x'], params['c_x'])
        y_pred = poly2(t_data, params['a_y'], params['b_y'], params['c_y'])
        z_pred = poly2(t_data, params['a_z'], params['b_z'], params['c_z'])

        sse_x = 0.5 * torch.sum((x_data - x_pred) ** 2)
        sse_y = 0.5 * torch.sum((y_data - y_pred) ** 2)
        sse_z = 0.5 * torch.sum((z_data - z_pred) ** 2)
        sse_total = sse_x + sse_y + sse_z

        pen_cov = covariance_penalty(x_pred, y_pred, z_pred, cov_target)

        pen_ang = angular_velocity_penalty_3d(t_data, params, omega_target=0.0, dt=1.0)

        total_loss = sse_total + lambda_cov * pen_cov + lambda_ang * pen_ang

        total_loss.backward()
        optimizer.step()

        ratio_cov = 0.01
        ratio_ang = 0.01
        if sse_total.item() > 1e-10:
            ratio_cov = (pen_cov.item()) / (sse_total.item())
            ratio_ang = (pen_ang.item()) / (sse_total.item())

        lambda_cov = lambda_cov * np.exp(reg_lr * (ratio_cov - target_ratio_cov))
        lambda_ang = lambda_ang * np.exp(reg_lr * (ratio_ang - target_ratio_ang))

        if epoch % 1000 == 0:
            print(f"Epoch {epoch} | Loss={total_loss.item():.6f} | SSE={sse_total.item():.6f} "
                  f"| Cov={pen_cov.item():.4f} (lambda={lambda_cov:.6f}, ratio={ratio_cov:.6f}) "
                  f"| Ang={pen_ang.item():.4f} (lambda={lambda_ang:.6f}, ratio={ratio_ang:.6f})")

    next_t = len(t_data_np) + 1
    a_x, b_x, c_x = params['a_x'].item(), params['b_x'].item(), params['c_x'].item()
    a_y, b_y, c_y = params['a_y'].item(), params['b_y'].item(), params['c_y'].item()
    a_z, b_z, c_z = params['a_z'].item(), params['b_z'].item(), params['c_z'].item()

    pred_x = poly2(next_t, a_x, b_x, c_x)
    pred_y = poly2(next_t, a_y, b_y, c_y)
    pred_z = poly2(next_t, a_z, b_z, c_z)

    # NEW: 分别计算各维度的 SSE
    sse_x_val = 0.5 * torch.sum((x_data - poly2(t_data, params['a_x'], params['b_x'], params['c_x'])) ** 2).item()
    sse_y_val = 0.5 * torch.sum((y_data - poly2(t_data, params['a_y'], params['b_y'], params['c_y'])) ** 2).item()
    sse_z_val = 0.5 * torch.sum((z_data - poly2(t_data, params['a_z'], params['b_z'], params['c_z'])) ** 2).item()

    print("\nFitted function relationships:")
    print(f"x(t) = {a_x:.6f} + {b_x:.6f} * t + {c_x:.6f} * t^2")
    print(f"y(t) = {a_y:.6f} + {b_y:.6f} * t + {c_y:.6f} * t^2")
    print(f"z(t) = {a_z:.6f} + {b_z:.6f} * t + {c_z:.6f} * t^2")
    print(f"SSE_x = {sse_x_val:.6f}")      # NEW: 打印x方向SSE
    print(f"SSE_y = {sse_y_val:.6f}")      # NEW: 打印y方向SSE
    print(f"SSE_z = {sse_z_val:.6f}")      # NEW: 打印z方向SSE
    print(f"SSE_total = {sse_total.item():.6f}")  # NEW: 打印总SSE

    print(f"\nPrediction for t = {next_t}:")
    print(f"Predicted position: x = {pred_x:.6f}, y = {pred_y:.6f}, z = {pred_z:.6f}")

    # -------------------------
    #  3D Plot
    # -------------------------
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(x_data_np, y_data_np, z_data_np, c='red', marker='o', label='Drone location')
    ax.plot(x_data_np, y_data_np, z_data_np, linestyle='--', color='black', label='Flight path')

    t_markers = np.arange(1, next_t + 1)
    x_markers = poly2(t_markers, a_x, b_x, c_x)
    y_markers = poly2(t_markers, a_y, b_y, c_y)
    z_markers = poly2(t_markers, a_z, b_z, c_z)
    ax.scatter(x_markers, y_markers, z_markers, c='green', marker='o', label='Estimated location')

    t_vals = np.linspace(1, next_t, 50)
    x_vals = poly2(t_vals, a_x, b_x, c_x)
    y_vals = poly2(t_vals, a_y, b_y, c_y)
    z_vals = poly2(t_vals, a_z, b_z, c_z)
    ax.plot(x_vals, y_vals, z_vals, c='green', label='Constant acceleration model')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Constant Acceleration Model (Adaptive Reg)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(fontsize=6)
    ax.view_init(azim=30)

    plt.tight_layout()
    plt.show()

# -----------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select main function to run.")
    parser.add_argument("--mode", type=str, default="original",
                        help="Mode to run: 'original' for original main, 'cov' for covariance and angular velocity validation.")
    args = parser.parse_args()

    if args.mode == "cov":
        print("Running covariance and angular velocity regression main...")
        main_cov_angle_reg()
    else:
        print("Running Decaying Learning Rate main...")
        main_Decaying_Learning_Rate()
