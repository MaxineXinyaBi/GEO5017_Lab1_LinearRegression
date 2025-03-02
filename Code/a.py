import numpy as np
from gradient_descent import gradient_descent

def constant_speed_gradient(params, timestamps, pos):
    """
    Parameters
    a0: initial position
    a1: speed
    timestamps: list of time
    pos: measured position at each time

    Returns:
    the gradient of a0 and a1
    """

    a0, a1 = params
    predicted = a0 + a1 * timestamps
    predict_error = pos - predicted
    derivative_a0 = -2 * np.sum(predict_error)
    derivative_a1 = -2 * np.sum(predict_error * timestamps)
    return np.array([derivative_a0, derivative_a1])

def constant_speed_residual(params, timestamps, pos):
    """
    Parameters:
    params: Optimized a0 and a1 value
    timestamps: list of time
    pos: measured position at each time

    Returns:
    residual error
    """
    a0, a1 = params
    predicted = a0 + a1 * timestamps
    error = predicted - pos
    return np.sum(error ** 2)

points = np.array([
    [2, 0, 1],
    [1.08, 1.68, 2.38],
    [-0.83, 1.82, 2.49],
    [-1.97, 0.28, 2.15],
    [-1.31, -1.51, 2.59],
    [0.57, -1.91, 4.32]
])

x_values = points[:, 0]
y_values = points[:, 1]
z_values = points[:, 2]
timestamps = np.array([1, 2, 3, 4, 5, 6])

print("Fitting X coordinates:")
x_params = gradient_descent([0,1],
                          lambda params: constant_speed_gradient(params, timestamps, x_values),
                          0.01, 1000, 0.001)
x_residual = constant_speed_residual(x_params, timestamps, x_values)
print(f"the final function on x axis is x_pos = {x_params[0]:.3f} + {x_params[1]:.3f} * t, and the residual is {x_residual:.3f}")


print("\nFitting Y coordinates:")
y_params = gradient_descent([0,1],
                          lambda params: constant_speed_gradient(params, timestamps, y_values),
                          0.01, 1000, 0.001)
y_residual = constant_speed_residual(y_params, timestamps, y_values)
print(f"the final function on y axis is y_pos = {y_params[0]:.3f} + {y_params[1]:.3f} * t, and the residual is {y_residual:.3f}")

print("\nFitting Z coordinates:")
z_params = gradient_descent([0,1],
                          lambda params: constant_speed_gradient(params, timestamps, z_values),
                          0.01, 1000, 0.001)
z_residual = constant_speed_residual(z_params, timestamps, z_values)
print(f"the final function on z axis is z_pos = {z_params[0]:.3f} + {z_params[1]:.3f} * t, and the residual is {z_residual:.3f}")