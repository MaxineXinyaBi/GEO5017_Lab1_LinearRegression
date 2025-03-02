import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
#  Model
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
    a = a
    b = b
    c = c

    loss_list = []
    epoch_list = []

    for epoch in range(epochs + 1):
        loss_val = sse_loss(t_data, x_data, a, b, c)
        if loss_val >8:
            break
        loss_list.append(loss_val)
        epoch_list.append(epoch)

        d_a, d_b, d_c = grad_sse(t_data, x_data, a, b, c)

        step_a = lr * d_a
        step_b = lr * d_b
        step_c = lr * d_c
        step_size = np.sqrt(step_a ** 2 + step_b ** 2 + step_c ** 2)

        if step_size < tolerance:
            break

        a = a - step_a
        b = b - step_b
        c = c - step_c

    print(f"Stop at epoch {epoch}, step_size={step_size:.8f}")
    return a, b, c, epoch_list, loss_list


# -------------------------
#  Testing
# -------------------------


if __name__ == "__main__":

    file_path = "metadata.csv"
    df = pd.read_csv(file_path)
    x_data = np.array(df["x"], dtype=float)
    t_data = np.arange(1, len(x_data) + 1, dtype=float)
    print("x_data:", x_data)
    print("t_data:", t_data)

    lr_list = [0.0001, 0.0005, 0.0008, 0.000845895]  # Learning rate (tunable)
    color_list = ['blue', 'orange', 'yellow', 'red']

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, lr in enumerate(lr_list):
        print(f"\n==== Learning Rate: {lr:.2e} ====")
        a_hat, b_hat, c_hat, ep, loss_list = gradient_descent_poly2(
            t_data, x_data, lr,
            epochs=60000,  # Max allowed iterations (tunable)
            tolerance=1e-6,  # Min allowed step size (tunable)
            a=0, b=0, c=0  # Initial polynomial parameters (tunable)
        )

        ax.plot(ep, loss_list,
                label=f"lr={lr:.1e}",
                color=color_list[i % len(color_list)])

        final_loss = sse_loss(t_data, x_data, a_hat, b_hat, c_hat)
        print(f"a = {a_hat:.6f}, b = {b_hat:.6f}, c = {c_hat:.6f}, "
              f"Final SSE = {final_loss:.6f}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (SSE)")
    ax.set_title("SSE vs. Epoch for Different LRs")
    ax.legend()

    plt.tight_layout()
    plt.show()
