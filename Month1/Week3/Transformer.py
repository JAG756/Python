import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]

# 2. 完整 Transformer
class TransformerModel(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=2, vocab_size=1000):
        super().__init__()
        self.d_model = d_model
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer 主干
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )
        
        # 最终分类层（预测下一个词）
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        # 编码输入
        src_emb = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        src_emb = self.pos_encoder(src_emb)
        
        # 解码输入
        tgt_emb = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        tgt_emb = self.pos_encoder(tgt_emb)
        
        # Transformer 前向
        out = self.transformer(src_emb, tgt_emb)
        
        # 预测下一个词
        return self.fc(out)

# ------------------- 测试一下 -------------------
if __name__ == '__main__':
    # 超参数
    vocab_size = 1000
    d_model = 128
    
    # 模型
    model = TransformerModel(d_model=d_model, nhead=8, num_layers=2, vocab_size=vocab_size)
    
    # 构造假数据：batch=8, seq_len=20
    src = torch.randint(0, vocab_size, (8, 20))   # 输入序列
    tgt = torch.randint(0, vocab_size, (8, 20))  # 目标序列
    
    # 前向
    out = model(src, tgt)
    print("输出形状:", out.shape)  # [8, 20, 1000] → 预测每个位置的词