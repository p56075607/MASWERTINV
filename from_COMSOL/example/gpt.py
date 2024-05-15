# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 生成一些不規則分布的散點數據
np.random.seed(0)
num_points = 100
x = np.random.uniform(0, 10, num_points)
y = np.random.uniform(0, 10, num_points)
z = np.sin(x) + np.cos(y)

# 定義網格
grid_x, grid_y = np.mgrid[0:10:100j, 0:10:100j]

# 使用 griddata 進行內插
grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

# 繪製原始散點數據
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(x, y, c=z, s=50, cmap='viridis')
plt.colorbar(label='z')
plt.title('Original Scatter Data')

# 繪製內插後的網格數據
plt.subplot(1, 2, 2)
plt.imshow(grid_z.T, extent=(0, 10, 0, 10), origin='lower', cmap='viridis')
plt.colorbar(label='z')
plt.title('Interpolated Grid Data')

plt.show()
