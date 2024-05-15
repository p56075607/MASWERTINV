# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# �ͦ��@�Ǥ��W�h���������I�ƾ�
np.random.seed(0)
num_points = 100
x = np.random.uniform(0, 10, num_points)
y = np.random.uniform(0, 10, num_points)
z = np.sin(x) + np.cos(y)

# �w�q����
grid_x, grid_y = np.mgrid[0:10:100j, 0:10:100j]

# �ϥ� griddata �i�椺��
grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

# ø�s��l���I�ƾ�
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(x, y, c=z, s=50, cmap='viridis')
plt.colorbar(label='z')
plt.title('Original Scatter Data')

# ø�s�����᪺����ƾ�
plt.subplot(1, 2, 2)
plt.imshow(grid_z.T, extent=(0, 10, 0, 10), origin='lower', cmap='viridis')
plt.colorbar(label='z')
plt.title('Interpolated Grid Data')

plt.show()
