
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


def load_model(fn, cfg):

    params = ModelParameters(
        d_model=cfg.get('d_model', 128),
        d_ff=cfg.get('d_ff', 512),
        n_enc_heads=cfg.get('n_enc_heads', 4),
        n_enc_layers=cfg.get('n_enc_layers', 2),
        n_dec_heads=cfg.get('n_dec_heads', 4),
        n_dec_layers=cfg.get('n_dec_layers', 2),
        n_mels=cfg.get('n_mels', 128),
        n_tokens=n_tokens,
        max_mel_length=cfg.get('max_mel_length', 512),
        max_sequence_length=cfg.get('max_sequence_length', 64),
        dropout=cfg.get('dropout'))



def foo():
    penc = PositionalEncoding(d_model=4, max_len=20)
    x = torch.randn(2, 12, 4)
    print(penc(x) - x)

if __name__ == '__main__':
    foo()
