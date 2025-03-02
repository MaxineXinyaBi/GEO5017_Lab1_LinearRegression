
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd


filename = "metadata.csv"
df = pd.read_csv(filename)
points = list(zip(df["x"].astype(float), df["y"].astype(float), df["z"].astype(float)))
print(points)


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(projection='3d')

xs = [p[0] for p in points]
ys = [p[1] for p in points]
zs = [p[2] for p in points]

ax.scatter(xs, ys, zs, c='red', marker='o', label='Drone location')
ax.plot(xs, ys, zs, linestyle='--', color='black', label='Flight path')

for i, (x, y, z) in enumerate(points, start=1):
    offset_x, offset_y, offset_z = 0.1, 0.1, 0.1
    ax.text(x + offset_x, y + offset_y, z + offset_z, str(i), color='blue')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.set_title('Drone Trajectory', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=8)
ax.legend(fontsize=6)
ax.view_init(azim=30)

plt.tight_layout()
plt.show()

