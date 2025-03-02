# -*- coding = utf-8 -*-
# @Time : 2025/3/1 2:57 PM
# @Author : 叶泓瑜
# @File : draw.py
# @Software : PyCharm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# 已有的无人机坐标点
points = [
    (2, 0, 1),
    (1.08, 1.68, 2.38),
    (-0.83, 1.82, 2.49),
    (-1.97, 0.28, 2.15),
    (-1.31, -1.51, 2.59),
    (0.57, -1.91, 4.32)
]

# 创建横向更宽的画布
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(projection='3d')

# 分离坐标
xs = [p[0] for p in points]
ys = [p[1] for p in points]
zs = [p[2] for p in points]

# Scatter plot with red circles (label for legend)
ax.scatter(xs, ys, zs, c='red', marker='o', label='Drone location')

# Connect the points in a black dotted line (label for legend)
ax.plot(xs, ys, zs, linestyle='--', color='black', label='Flight path')

"""
# 标注编号，编号4与5在点左侧
for i, (x, y, z) in enumerate(points, start=1):
    offset_x, offset_y, offset_z = 0.1, 0.1, 0.1
    if i in [1, 2]:
        offset_y = 0.2  # 向左偏移
    ax.text(x + offset_y, y + offset_y, z + offset_z, str(i), color='blue')

"""

# 在 t ∈ [0, 6] 取值范围内计算 x, y, z
t_vals = np.linspace(1, 6, 50)  # 可根据需要调整取样数量
x_vals = 1.424 - 0.431 * t_vals
y_vals = 2.081 - 0.580 * t_vals
z_vals = 0.754 + 0.493 * t_vals

# 在 t=1,2,3,4,5,6 的位置标注绿色圆点
t_markers = np.array([1, 2, 3, 4, 5, 6])
x_markers = 1.424 - 0.431 * t_markers
y_markers = 2.081 - 0.580 * t_markers
z_markers = 0.754 + 0.493 * t_markers
ax.scatter(x_markers, y_markers, z_markers, c='green', marker='o', label='Estimate location')

# 将 (x, y, z) 在 [0, 6] 上的取值绘制在同一张图中
ax.plot(x_vals, y_vals, z_vals, c='green', label='Constant Speed Model')



# 坐标轴标签
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# 图标题
ax.set_title('Constant Speed Model', fontsize=14)
# ax.set_title('Drone Trajectory', fontsize=14)

# 调整坐标刻度字体大小
ax.tick_params(axis='both', which='major', labelsize=8)

# 添加图例（并缩小字体）
ax.legend(fontsize=6)
ax.view_init(azim=30)

# 防止标签或图例被截断
plt.tight_layout()
plt.show()

