
import math
import torch


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len, dropout=0.0):
        super().__init__()

        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # assume x shape is (batch, t, dim)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def load_transformer(fn, params, device=None):
    # simple helper function to load a saved transformer
    from .transformer import Transformer
    model = Transformer(params)
    model.load_state_dict(torch.load(fn, map_location=device))
    return model


def model_size(model):
    num_params = sum(param.numel() for param in model.parameters())
    return num_params


