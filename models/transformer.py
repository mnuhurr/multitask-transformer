
import torch
import torch.nn.functional as F

from dataclasses import dataclass

from .utils import PositionalEncoding


@dataclass(frozen=True)
class ModelParameters:
    n_tokens: int
    d_model: int = 128
    d_ff: int = 512
    n_enc_heads: int = 2
    n_enc_layers: int = 2
    n_dec_heads: int = 2
    n_dec_layers: int = 2
    n_mels: int = 128
    max_mel_length: int = 512
    max_sequence_length: int = 64
    dropout: float = 0.1


class Encoder(torch.nn.Module):
    def __init__(self, n_mels, max_mel_length, d_model, d_ff, n_enc_heads, n_enc_layers, dropout=0.1):
        super().__init__()

        self.conv_net = torch.nn.Sequential(
            torch.nn.Conv1d(n_mels, d_model, kernel_size=5, padding=2),
            torch.nn.GELU(),
            torch.nn.Dropout(p=dropout),

            #torch.nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, stride=2),
            #torch.nn.GELU(),
            #torch.nn.Dropout(p=dropout),

            #torch.nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, stride=2),
            #torch.nn.GELU(),
            #torch.nn.Dropout(p=dropout),

            torch.nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, stride=2),
            torch.nn.GELU(),
        )

        self.conv_ln = torch.nn.LayerNorm(d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_mel_length)

        enc_layer = torch.nn.TransformerEncoderLayer(d_model, n_enc_heads, d_ff, dropout, batch_first=True, norm_first=True)
        self.encoder = torch.nn.TransformerEncoder(enc_layer, n_enc_layers)

        self.layernorm = torch.nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = self.conv_net(x)

        x = x.permute(0, 2, 1)
        x = self.conv_ln(x)

        # use pooling to make the mask the same length
        if mask is not None:
            #mask = F.max_pool1d(mask.unsqueeze(1), kernel_size=3, padding=1, stride=2)
            #mask = F.max_pool1d(mask, kernel_size=3, padding=1, stride=2)
            #mask = F.max_pool1d(mask, kernel_size=3, padding=1, stride=2)[:, 0, :]
            mask = F.max_pool1d(mask.unsqueeze(1), kernel_size=3, padding=1, stride=2)[:, 0, :]

        x = self.positional_encoding(x)
           
        x = self.encoder(x, src_key_padding_mask=mask)

        return self.layernorm(x), mask


class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, max_seq_length, d_model, d_ff, n_dec_heads, n_dec_layers, dropout=0.1):
        super().__init__()

        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)

        #self.positional_embedding = torch.nn.Parameter(torch.empty(max_seq_length, d_model))
        self.positional_embedding = torch.nn.Parameter(0.1 * torch.randn(max_seq_length, d_model))
        mask = torch.empty(max_seq_length, max_seq_length).fill_(-float('inf')).triu(1)
        self.register_buffer('mask', mask.to(torch.bool), persistent=False)

        dec_layer = torch.nn.TransformerDecoderLayer(d_model, n_dec_heads, d_ff, dropout, batch_first=True, norm_first=True)
        self.decoder = torch.nn.TransformerDecoder(dec_layer, n_dec_layers)

        self.layernorm = torch.nn.LayerNorm(d_model)

    def forward(self, tokens, enc_out, enc_mask, token_mask):
        seq_len = tokens.size(-1)
        x = self.token_embedding(tokens)
        x = x + self.positional_embedding[:seq_len, :]

        x = self.decoder(x, enc_out, tgt_mask=self.mask[:seq_len, :seq_len], tgt_key_padding_mask=token_mask, memory_key_padding_mask=enc_mask)

        x = self.layernorm(x)
        logits = x @ torch.transpose(self.token_embedding.weight, 0, 1)

        return logits


class Transformer(torch.nn.Module):
    def __init__(self, params: ModelParameters):
        super().__init__()

        self.encoder = Encoder(
            n_mels=params.n_mels, 
            max_mel_length=params.max_mel_length, 
            d_model=params.d_model,
            d_ff=params.d_ff, 
            n_enc_heads=params.n_enc_heads, 
            n_enc_layers=params.n_enc_layers, 
            dropout=params.dropout)

        self.decoder = Decoder(
            vocab_size=params.n_tokens,
            max_seq_length=params.max_sequence_length,
            d_model=params.d_model,
            d_ff=params.d_ff,
            n_dec_heads=params.n_dec_heads,
            n_dec_layers=params.n_dec_layers,
            dropout=params.dropout)

    def forward(self, mels, tokens, mel_mask, token_mask):
        enc_out, enc_mask = self.encoder(mels, mel_mask)
        logits = self.decoder(tokens, enc_out, enc_mask, token_mask)

        return logits

