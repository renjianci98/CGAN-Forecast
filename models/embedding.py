import torch
import torch.nn as nn

from .models import register


@register('Embedding')
class Embedding(nn.Module):
    def __init__(self, input_dim,embed_dim):
        super(Embedding, self).__init__()
        self.conv_layer = nn.Conv1d(input_dim, embed_dim,
                               kernel_size=3, padding='same')

    def forward(self, X):
        '''
        args:
            X - 需要预测的时间序列，其形状为(batch_size,seq_len,feature_dim)
        '''
        X = X.permute(0, 2, 1)  # 转换后两个维度以适应一维卷积
        X = self.conv_layer(X).permute(0, 2, 1)  # 转回
        X = X.permute(1, 0, 2)  # 转换前两个维度以适应positional encoding
        pos_encoder = PositionalEncoding(X.shape[2], X.shape[0])
        X = pos_encoder(X)
        return X


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seqlen):
        # d_model是每个词embedding后的维度
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros((seqlen, d_model),device='cuda')
        position = torch.arange(0, seqlen, dtype=torch.float).unsqueeze(1)
        div_term2 = torch.pow(torch.tensor(10000.0),
                              torch.arange(0, d_model, 2).float()/d_model)
        div_term1 = torch.pow(torch.tensor(10000.0),
                              torch.arange(1, d_model, 2).float()/d_model)
        # 高级切片方式，即从0开始，两个步长取一个。即奇数和偶数位置赋值不一样。直观来看就是每一句话的
        pe[:, 0::2] = torch.sin(position * div_term2)
        pe[:, 1::2] = torch.cos(position * div_term1)
        # 这里是为了与x的维度保持一致，释放了一个维度
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
