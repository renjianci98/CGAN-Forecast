import torch
import torch.nn as nn

from .models import register
from . import models


@register('Generator_Transformer')
class Generator(nn.Module):
    '''
    CGAN中的generator，其中编解码器可以套用不同模型

    '''

    def __init__(self, encoder, decoder,embedding,predict_len):
        super(Generator, self).__init__()
        self.embedding=models.make('Embedding',**embedding['args'])
        self.encoder = models.make(encoder['name'], **encoder['args'])
        self.fcblock1 = models.make('FC_Block', input_dim=128, K=3, M=128)
        self.fcblock2 = models.make('FC_Block', input_dim=256, K=3, M=128)
        self.decoder = models.make(decoder['name'], **decoder['args'],predict_len=predict_len)
        self.fc = nn.Linear(128, 2)
        self.sigmoid = nn.Sigmoid()
        self.predict_len=predict_len

    def forward(self, X, Z):
        '''
        args:
            X - 需要预测的时间序列，其形状为(batch_size,seq_len,feature_dim)
            Z - 生成器初始随机噪声，其形状为(batch_size,feature_dim),feature_dim默认128维
        '''
        
        X= self.embedding(X)
        token=X
        X = self.encoder(X)
        Z = self.fcblock1(Z)
        X = torch.concat((X, Z), dim=2)
        X = self.fcblock2(X)
        X = self.decoder(X,token)
        X=X[self.predict_len:,:,:]
        X = X.transpose(0, 1)
        X = self.fc(X)
        forecast = self.sigmoid(X)
        return forecast


@register('Discriminator_Transformer')
class Discriminator(nn.Module):
    '''
    CGAN中的discriminator，其中编码器可以套用不同模型

    '''

    def __init__(self, encoder_x, encoder_y):
        super(Discriminator, self).__init__()
        self.encoder_x = models.make(encoder_x['name'], **encoder_x['args'])
        self.encoder_y = models.make(encoder_y['name'], **encoder_y['args'])
        self.fcblock1 = models.make('FC_Block', input_dim=128, K=3, M=128)
        self.fcblock2 = models.make('FC_Block', input_dim=256, K=3, M=128)
        self.linear = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, Y):
        '''
        args:
            X - 需要预测的时间序列，其形状为(batch_size,seq_len,feature_dim)
            Y - GT或生成器生成的预测结果，其形状为(batch_size,seq_len,feature_dim)
        '''
        X = self.encoder_x(X)
        X = self.fcblock1(X)
        Y = self.encoder_y(Y)
        X = torch.concat((X, Y), dim=1)
        X = self.fcblock2(X)
        X = self.linear(X)
        score = self.sigmoid(X)
        return score

