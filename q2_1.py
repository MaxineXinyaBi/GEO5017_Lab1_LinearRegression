import matplotlib.pyplot as plt
import numpy as np

# plot trajectory
points = np.array([
    [2, 0, 1],
    [1.08, 1.68, 2.38],
    [-0.83, 1.82, 2.49],
    [-1.97, 0.28, 2.15],
    [-1.31, -1.51, 2.59],
    [0.57, -1.91, 4.32]
])

fig_1 = plt.figure(figsize = (12, 12))
ax_1 = fig_1.add_subplot(projection = '3d')

x_values = points[:, 0]
y_values = points[:, 1]
z_values = points[:, 2]

ax_1.scatter(x_values, y_values, z_values, c='b', marker='o', linewidths=1.5, label='Waypoints')
ax_1.set_xlabel('X')
ax_1.set_ylabel('Y')
ax_1.set_zlabel('Z')

plt.plot(x_values, y_values, z_values, c='r',linestyle='--',label='Flight Path')

plt.title("Drone Trajectory")
ax_1.legend()

plt.show()






