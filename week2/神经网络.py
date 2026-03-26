import numpy as np

# ====================== 1. 定义激活函数 ======================
# ReLU 激活函数：隐藏层用
def relu(x):
    return np.maximum(0, x)

# ReLU 的导数（反向传播用）
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Sigmoid：输出层二分类用
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ====================== 2. 构建数据集 ======================
# 简单二分类任务：输入2个特征，输出0/1
X = np.array([[0,0], [0,1], [1,0], [1,1]])  # 输入
y = np.array([[0], [1], [1], [0]])          # 真实标签（异或问题）

# ====================== 3. 初始化网络参数（层） ======================
# 网络结构：输入2维 → 隐藏层4个神经元 → 输出1维
input_size = 2    # 输入层维度
hidden_size = 4   # 隐藏层神经元个数
output_size = 1   # 输出层维度

# 权重 W 和偏置 b（随机初始化）
W1 = np.random.randn(input_size, hidden_size)  # 输入→隐藏层权重
b1 = np.zeros((1, hidden_size))                # 隐藏层偏置
W2 = np.random.randn(hidden_size, output_size) # 隐藏→输出层权重
b2 = np.zeros((1, output_size))                # 输出层偏置

# 训练参数
learning_rate = 0.1  # 学习率
epochs = 10000       # 训练轮数

# ====================== 4. 训练：前向传播 + 反向传播 ======================
for epoch in range(epochs):
    # ---------------- 前向传播（数据从前往后算） ----------------
    # 第一层：输入 → 隐藏层
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)  # 激活函数

    # 第二层：隐藏层 → 输出层
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)  # 输出概率

    # 计算损失（误差）
    loss = np.mean((a2 - y) ** 2)

    # ---------------- 反向传播（误差从后往前算） ----------------
    # 输出层梯度
    dz2 = (a2 - y) * a2 * (1 - a2)  # sigmoid导数
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    # 隐藏层梯度
    dz1 = np.dot(dz2, W2.T) * relu_derivative(z1)
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # ---------------- 更新权重（学习的核心） ----------------
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # 每1000轮打印损失
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# ====================== 5. 测试结果 ======================
print("\n==== 预测结果 ====")
print("输入：[0,0] [0,1] [1,0] [1,1]")
print("预测输出：", a2.round(0).flatten())
print("真实标签：", y.flatten())
