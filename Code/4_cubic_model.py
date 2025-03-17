import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def poly3(t, a, b, c, d):
    return a + b*t + c*(t**2) + d*(t**3)

def sse_loss(t_data, x_data, a, b, c, d):
    errors = x_data - poly3(t_data, a, b, c, d)
    return 0.5 * np.sum(errors ** 2)

def grad_sse(t_data, x_data, a, b, c, d):
    errors = x_data - poly3(t_data, a, b, c, d)
    d_a = -np.sum(errors)
    d_b = -np.sum(errors * t_data)
    d_c = -np.sum(errors * (t_data ** 2))
    d_d = -np.sum(errors * (t_data ** 3))
    return d_a, d_b, d_c, d_d

def gradient_descent_poly3(t_data, x_data, lr, epochs, tolerance, a, b, c, d):
    a_cache, b_cache, c_cache, d_cache = a, b, c, d
    a_prev, b_prev, c_prev, d_prev = 0, 0, 0, 0
    loss_list = []
    epoch_list = []
    epoch_val = 0
    count = 0
    prev_loss = float('inf')
    while epoch_val < epochs:
        count += 1
        a_prev, b_prev, c_prev, d_prev = a_cache, b_cache, c_cache, d_cache
        d_a, d_b, d_c, d_d = grad_sse(t_data, x_data, a_cache, b_cache, c_cache, d_cache)
        step_a = lr * d_a
        step_b = lr * d_b
        step_c = lr * d_c
        step_d = lr * d_d
        step_size = np.sqrt(step_a**2 + step_b**2 + step_c**2 + step_d**2)
        a_new = a_cache - step_a
        b_new = b_cache - step_b
        c_new = c_cache - step_c
        d_new = d_cache - step_d
        current_loss = sse_loss(t_data, x_data, a_new, b_new, c_new, d_new)
        if current_loss < prev_loss:
            a_cache, b_cache, c_cache, d_cache = a_new, b_new, c_new, d_new
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
            a_cache, b_cache, c_cache, d_cache = a_prev, b_prev, c_prev, d_prev
    print("Final iteration:", count)
    return a_cache, b_cache, c_cache, d_cache, epoch_list, loss_list

def main_Decaying_Learning_Rate():
    file_path = "metadata.csv"
    df = pd.read_csv(file_path)
    x_data = np.array(df["x"], dtype=float)
    y_data = np.array(df["y"], dtype=float)
    z_data = np.array(df["z"], dtype=float)
    t_data = np.arange(1, len(x_data) + 1, dtype=float)
    init_lr = 0.01      # 初始学习率（可调）
    max_epochs = 10000000  # 最大迭代次数（可调）
    tolerance = 1e-6    # 最小步长（可调）
    a, b, c, d = 0, 0, 0, 0
    print("\nFitting x(t):")
    a_x, b_x, c_x, d_x, epochs_x, loss_x = gradient_descent_poly3(
        t_data, x_data, init_lr, max_epochs, tolerance, a, b, c, d
    )
    sse_x = sse_loss(t_data, x_data, a_x, b_x, c_x, d_x)
    print("\nFitting y(t):")
    a_y, b_y, c_y, d_y, epochs_y, loss_y = gradient_descent_poly3(
        t_data, y_data, init_lr, max_epochs, tolerance, a, b, c, d
    )
    sse_y = sse_loss(t_data, y_data, a_y, b_y, c_y, d_y)
    print("\nFitting z(t):")
    a_z, b_z, c_z, d_z, epochs_z, loss_z = gradient_descent_poly3(
        t_data, z_data, init_lr, max_epochs, tolerance, a, b, c, d
    )
    sse_z = sse_loss(t_data, z_data, a_z, b_z, c_z, d_z)
    total_SSE = sse_x + sse_y + sse_z
    print("\nFitted function relationships:")
    print(f"x(t) = {a_x:.6f} + {b_x:.6f} * t + {c_x:.6f} * t^2 + {d_x:.6f} * t^3, SSE_x = {sse_x:.6f}")
    print(f"y(t) = {a_y:.6f} + {b_y:.6f} * t + {c_y:.6f} * t^2 + {d_y:.6f} * t^3, SSE_y = {sse_y:.6f}")
    print(f"z(t) = {a_z:.6f} + {b_z:.6f} * t + {c_z:.6f} * t^2 + {d_z:.6f} * t^3, SSE_z = {sse_z:.6f}")
    print(f"Total SSE = {total_SSE:.6f}")
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_data, y_data, z_data, c='red', marker='o', label='Drone location')
    ax.plot(x_data, y_data, z_data, linestyle='--', color='black', label='Flight path')
    t_markers = np.arange(1, len(x_data) + 1)
    x_markers = poly3(t_markers, a_x, b_x, c_x, d_x)
    y_markers = poly3(t_markers, a_y, b_y, c_y, d_y)
    z_markers = poly3(t_markers, a_z, b_z, c_z, d_z)
    ax.scatter(x_markers, y_markers, z_markers, c='green', marker='o', label='Estimated location')
    t_vals = np.linspace(1, len(x_data), 50)
    x_vals = poly3(t_vals, a_x, b_x, c_x, d_x)
    y_vals = poly3(t_vals, a_y, b_y, c_y, d_y)
    z_vals = poly3(t_vals, a_z, b_z, c_z, d_z)
    ax.plot(x_vals, y_vals, z_vals, c='green', label='Cubic model')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Cubic Polynomial Model (Decaying LR)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(fontsize=6)
    ax.view_init(azim=30)
    plt.tight_layout()
    plt.show()

    t_plot = np.linspace(1, 6, 200)
    x_fit = poly3(t_plot, a_x, b_x, c_x, d_x)
    y_fit = poly3(t_plot, a_y, b_y, c_y, d_y)
    z_fit = poly3(t_plot, a_z, b_z, c_z, d_z)
    mask_06 = (t_data >= 1) & (t_data <= 6)
    t_data_06 = t_data[mask_06]
    x_data_06 = x_data[mask_06]
    y_data_06 = y_data[mask_06]
    z_data_06 = z_data[mask_06]

    plt.figure()
    plt.title("Fitting of x(t)")
    plt.scatter(t_data_06, x_data_06, color='red', marker='o', label='metadata')
    plt.plot(t_plot, x_fit, color='blue', label='Fitted x(t)')
    plt.legend()
    plt.figure()
    plt.title("Fitting of y(t)")
    plt.scatter(t_data_06, y_data_06, color='red', marker='o', label='metadata')
    plt.plot(t_plot, y_fit, color='blue', label='Fitted y(t)')
    plt.legend()
    plt.figure()
    plt.title("Fitting of z(t)")
    plt.scatter(t_data_06, z_data_06, color='red', marker='o', label='metadata')
    plt.plot(t_plot, z_fit, color='blue', label='Fitted z(t)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select main function to run.")
    parser.add_argument("--mode", type=str, default="original", help="Mode to run: 'original' for decaying LR main (cubic).")
    args = parser.parse_args()
    print("Running Decaying Learning Rate main (cubic)...")
    main_Decaying_Learning_Rate()
