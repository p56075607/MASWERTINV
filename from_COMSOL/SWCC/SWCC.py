import numpy as np
import pandas as pd
from scipy.optimize import root
import matplotlib.pyplot as plt

# 定義 Van Genuchten 模型
def van_genuchten(h, theta_r, theta_s, alpha, n):
    m = 1 - 1/n
    theta = theta_r + (theta_s - theta_r) / (1 + (alpha * np.abs(h))**n)**m
    return theta

# 定義 Van Genuchten 模型的反函數
def van_genuchten_inv(theta, theta_r, theta_s, alpha, n):
    if theta == theta_s:
        return 0  # 飽和狀態下的壓力水頭設置為零
    m = 1 - 1/n
    func = lambda h: theta_r + (theta_s - theta_r) / (1 + (alpha * np.abs(h))**n)**m - theta
    
    # 使用多個初始猜測值來提高穩健性
    initial_guesses = [-1, -10, -100, -1000]
    for h_guess in initial_guesses:
        sol = root(func, h_guess)
        if sol.success:
            return sol.x[0]
    
    raise ValueError(f"Solution not found for theta = {theta}")

# [Soil] Van Genuchten 模型參數
theta_r = 0.034  # 殘餘含水量
theta_s = 0.46  # 飽和含水量
alpha = 1.6     # 經驗參數
n = 1.37       # 經驗參數

# [Rock] Van Genuchten 模型參數
theta_r = 0.031  # 殘餘含水量
theta_s = 0.467  # 飽和含水量
alpha = 3.64     # 經驗參數
n = 1.121       # 經驗參數

# 讀取 CSV 文件
input_csv_path = 'input.csv' 
output_csv_path = 'Hp_soil.csv'

df = pd.read_csv(input_csv_path)

# 確保水含量數據在有效範圍內
df = df[(df['water_content'] >= theta_r) & (df['water_content'] <= theta_s)]

# 計算壓力水頭
df['pressure_head'] = df['water_content'].apply(lambda theta: van_genuchten_inv(theta, theta_r, theta_s, alpha, n))

# 將結果保存為新的 CSV 文件
df.to_csv(output_csv_path, index=False)
print(f"Processed data saved to {output_csv_path}")

# 繪製壓力水頭分布圖
plt.figure(figsize=(10, 6))
sc = plt.scatter(df['X'], df['Y'], c=df['pressure_head'], cmap='viridis', marker='o')
plt.colorbar(sc, label='Pressure head (h)')
plt.clim(-18, 0) 
plt.ylabel('Y')
plt.show()
