import torch
import torch.nn as nn

from .models import register


@register('LSTM')
class LSTM(nn.Module):
    def __init__(self, feature_dim, n_hidden, n_layers, bidirectional, transposed=False) -> None:
        super(LSTM, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.transposed = transposed
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=n_hidden,
                            num_layers=n_layers, bidirectional=self.bidirectional)

    def forward(self, X):
        if self.transposed:
            input = X
        else:
            input = X.transpose(0, 1)
        batch_size = input.shape[1]
        h0 = torch.randn((2 if self.bidirectional else 1) *
                         self.n_layers, batch_size, self.n_hidden)
        c0 = torch.randn((2 if self.bidirectional else 1) *
                         self.n_layers, batch_size, self.n_hidden)
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()
        output, _ = self.lstm(input, (h0, c0))
        return output


@register('LSTM_Block')
class LSTMBlock(nn.Module):
    def __init__(self, feature_dim, n_hidden, output_dim, Q) -> None:
        super(LSTMBlock, self).__init__()
        self.output_dim = output_dim
        self.LSTM_start = LSTM(feature_dim, n_hidden, 1, True)
        self.LSTM_middle = LSTM(
            n_hidden*2, n_hidden, Q, True, transposed=True)
        self.LSTM_last = LSTM(n_hidden*2, n_hidden,
                              1, True, transposed=True)
        self.FC = nn.Linear(n_hidden, output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        X = self.LSTM_start(X)
        X_t = X
        X = self.LSTM_middle(X)
        X = X+X_t
        X = self.LSTM_last(X)
        shape = X.shape
        X = X.reshape(shape[0], shape[1], 2, -1)
        X = X[-1, :, 0, :]+X[0, :, 1, :]
        X = self.FC(X)
        X = self.relu(X)
        return X


