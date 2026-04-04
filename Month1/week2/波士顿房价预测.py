# ======================
# 完美修复版：波士顿房价预测
# 解决：无单位、无刻度、真实值不显示问题
# ======================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 修复绘图 backend，不弹窗报错
import matplotlib
matplotlib.use('TkAgg')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 评估
print("="*50)
print("模型评估结果")
print("="*50)
print(f"均方误差 (MSE): {mean_squared_error(y_test, y_pred):.2f}")
print(f"决定系数 (R²): {r2_score(y_test, y_test):.2f}")

print("\n过拟合检查")
print(f"训练集分数: {model.score(X_train, y_train):.2f}")
print(f"测试集分数: {model.score(X_test, y_test):.2f}")

# ======================
# ✅ 完美绘图修复（有刻度、有单位、有参考线）
# ======================
plt.figure(figsize=(9,7))

# 画散点（预测效果）
plt.scatter(y_test, y_pred, s=30, color='blue', alpha=0.6, label='预测值')

# 画完美预测参考线（红色虚线）
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='完美预测线')

# 加坐标轴名称 + 单位
plt.xlabel('真实房价（单位：千美元）', fontsize=12)
plt.ylabel('预测房价（单位：千美元）', fontsize=12)
plt.title('波士顿房价预测：真实值 vs 预测值', fontsize=14)

# 显示网格 + 刻度 + 图例
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.show()