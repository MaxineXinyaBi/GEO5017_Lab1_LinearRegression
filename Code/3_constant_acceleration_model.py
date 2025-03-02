
import numpy as np
import matplotlib.pyplot as plt


# -------------------------
#  1) 定义模型和损失函数
# -------------------------
def poly2(t, a, b, c):
    """
    二次多项式模型: f(t) = a + b*t + c*t^2
    """
    return a + b * t + c * (t ** 2)


def sse_loss(t_data, x_data, a, b, c):
    """
    计算 Sum of Squared Errors 的一半:
        SSE = 0.5 * sum( (x_i - f(t_i))^2 )
    """
    errors = x_data - poly2(t_data, a, b, c)
    return 0.5 * np.sum(errors ** 2)


# -------------------------
#  2) 计算梯度
# -------------------------
def grad_sse(t_data, x_data, a, b, c):
    """
    计算对 (a, b, c) 的偏导数:
        d(SSE)/da, d(SSE)/db, d(SSE)/dc
    """
    errors = x_data - poly2(t_data, a, b, c)  # 形状: (n,)
    # SSE = 0.5 * sum(errors^2)，
    # d(SSE)/da = - sum( errors * d(f)/da ) = -sum( errors * 1 )
    # d(SSE)/db = - sum( errors * t )
    # d(SSE)/dc = - sum( errors * t^2 )
    d_a = - np.sum(errors)
    d_b = - np.sum(errors * t_data)
    d_c = - np.sum(errors * (t_data ** 2))
    return d_a, d_b, d_c


# -------------------------
#  3) 梯度下降
# -------------------------
def gradient_descent_poly2(t_data, x_data, lr, epochs, verbose=True):
    """
    用梯度下降寻找 a, b, c 使 SSE 最小
    t_data, x_data: 输入数据
    lr: 学习率
    epochs: 迭代次数
    verbose: 是否打印训练过程
    """
    # 随机初始化 a, b, c
    a = 0
    b = 0
    c = 0

    # 记录损失随 epoch 的变化
    loss_list = []
    epoch_list = []

    for epoch in range(epochs + 1):
        # 1. 计算当前的 SSE
        loss_val = sse_loss(t_data, x_data, a, b, c)
        loss_list.append(loss_val)
        epoch_list.append(epoch)

        # 每隔 200 个 epoch 打印一次信息
        if verbose and (epoch % 200 == 0):
            print(f"Epoch {epoch:4d} | SSE = {loss_val:.6f} | "
                  f"a={a:.4f} b={b:.4f} c={c:.4f}")

        # 2. 计算梯度
        d_a, d_b, d_c = grad_sse(t_data, x_data, a, b, c)

        # 3. 参数更新 (梯度下降)
        a = a - lr * d_a
        b = b - lr * d_b
        c = c - lr * d_c

    # 最终参数
    return a, b, c, epoch_list, loss_list


# -------------------------
#  演示：使用给定的 X_data
# -------------------------
if __name__ == "__main__":
    # 题目中的时间点和 X_data
    t_data = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    x_data = np.array([2, 1.08, -0.83, -1.97, -1.31, 0.57], dtype=float)

    # 调用梯度下降来拟合二次多项式
    a_hat, b_hat, c_hat, ep, loss_list = gradient_descent_poly2(
        t_data, x_data,
        lr=0.0008,
        epochs=100000,
        verbose=True
    )

    print("\n===== 拟合结果 =====")
    print(f"a = {a_hat:.6f}, b = {b_hat:.6f}, c = {c_hat:.6f}")
    final_loss = sse_loss(t_data, x_data, a_hat, b_hat, c_hat)
    print(f"Final SSE = {final_loss:.6f}")

    # 画出损失随迭代变化
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(ep, loss_list, label='SSE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (SSE)')
    plt.title('Loss vs. Epoch')
    plt.legend()

    # 画出原始数据点 & 拟合曲线
    plt.subplot(1, 2, 2)
    plt.scatter(t_data, x_data, color='blue', label='data')
    t_lin = np.linspace(min(t_data), max(t_data), 100)
    x_fit = poly2(t_lin, a_hat, b_hat, c_hat)
    plt.plot(t_lin, x_fit, color='red', label='fitted curve')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Data & Fitted Polynomial')
    plt.legend()

    plt.tight_layout()
    plt.show()
