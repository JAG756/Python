import torch
import torch.nn as nn

# --------------------- 1. 准备数据 ---------------------
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # 输入
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])  # 真实值（y=2x+1）

# --------------------- 2. 定义模型 ---------------------
model = nn.Linear(1, 1)  # 线性模型 y = wx + b

# --------------------- 3. 定义优化器和损失 ---------------------
criterion = nn.MSELoss()       # 损失：计算误差
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 优化器：改权重

# --------------------- 4. 训练 ---------------------
for epoch in range(100):
    # 前向传播
    y_pred = model(x)

    # 计算损失
    loss = criterion(y_pred, y)

    # 反向传播 + 更新权重
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# --------------------- 5. 看结果 ---------------------
print("\n训练完成！")
print("w =", model.weight.item())
print("b =", model.bias.item())