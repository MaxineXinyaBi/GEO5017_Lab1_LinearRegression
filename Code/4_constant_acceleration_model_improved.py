import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------------------
#  Model (Decaying Learning Rate)
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
            a_cache, b_cache, c_cache = a_prev, b_prev, c_prev
    print(f"Epoch: {epoch_val + 1}; Lr: {lr:.1e}")
    return a_cache, b_cache, c_cache, epoch_list, loss_list


# -------------------------
#  Testing and Prediction
# -------------------------


if __name__ == "__main__":
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

    print("Fitting x(t):")
    a_x, b_x, c_x, epochs_x, loss_x = gradient_descent_poly2(t_data, x_data, init_lr, max_epochs, tolerance, a, b, c)
    sse_x = sse_loss(t_data, x_data, a_x, b_x, c_x)

    print("Fitting y(t):")
    a_y, b_y, c_y, epochs_y, loss_y = gradient_descent_poly2(t_data, y_data, init_lr, max_epochs, tolerance, a, b, c)
    sse_y = sse_loss(t_data, y_data, a_y, b_y, c_y)

    print("Fitting z(t):")
    a_z, b_z, c_z, epochs_z, loss_z = gradient_descent_poly2(t_data, z_data, init_lr, max_epochs, tolerance, a, b, c)
    sse_z = sse_loss(t_data, z_data, a_z, b_z, c_z)

    print("\nFitted function relationships:")
    print(f"x(t) = {a_x:.6f} + {b_x:.6f} * t + {c_x:.6f} * t^2, SSE_x = {sse_x:.6f}")
    print(f"y(t) = {a_y:.6f} + {b_y:.6f} * t + {c_y:.6f} * t^2, SSE_y = {sse_y:.6f}")
    print(f"z(t) = {a_z:.6f} + {b_z:.6f} * t + {c_z:.6f} * t^2, SSE_z = {sse_z:.6f}")

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
