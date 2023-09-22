import torch
import torch.nn as nn

from .models import register


@register('Trans_Encoder')
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_head, n_layers, d_forward) -> None:
        super(TransformerEncoder, self).__init__()
        self.n_layers = n_layers
        self.norm=nn.LayerNorm(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_forward)
        

    def forward(self, X):

        transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=self.n_layers,norm=self.norm)
        X = transformer_encoder(X)
        return X


@register('Trans_Decoder')
class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_head, n_layers, d_forward,predict_len) -> None:
        super(TransformerDecoder, self).__init__()
        self.n_layers=n_layers
        self.norm=nn.LayerNorm(d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_forward)
        self.predict_len=predict_len

    def forward(self, X,token_input):
        transformer_decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer, num_layers=self.n_layers,norm=self.norm)
        decoder_input=torch.zeros((self.predict_len,token_input.shape[1],token_input.shape[2]),device='cuda')
        decoder_input=torch.concat((token_input,decoder_input),dim=0)
        X = transformer_decoder(decoder_input,X)
        return X
