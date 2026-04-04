import torch
import torch.nn as nn

# ====================== 1. 定义 RNN 模型 ======================
class SimpleRNN(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, num_classes=2):
        super(SimpleRNN, self).__init__()
        
        # RNN 核心层
        self.rnn = nn.RNN(
            input_size=input_size,    # 每个时刻输入多少特征
            hidden_size=hidden_size,  # 隐藏层大小（记忆单元数）
            num_layers=2,             # 堆叠几层 RNN
            batch_first=True          # 输入形状: (batch, seq, feature)
        )
        
        # 最后用全连接层输出分类结果
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 前向传播：数据过 RNN
        # out: 每一步的输出   hn: 最后时刻的隐藏状态
        out, hn = self.rnn(x)  
        
        # 我们只需要**最后一步**的输出做分类
        out = self.fc(out[:, -1, :])  
        return out

# ====================== 2. 创建模型 ======================
model = SimpleRNN(input_size=5, hidden_size=16, num_classes=2)

# ====================== 3. 构造假数据测试 ======================
# 输入形状：(batch=4, seq_len=10, input_size=5)
test_input = torch.randn(4, 10, 5)  
output = model(test_input)

print("输入形状:", test_input.shape)
print("输出形状:", output.shape)  # 输出 (4,2) → 4个样本，2分类