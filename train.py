
import torch
import torch.nn.functional as F

import time
import math
from pathlib import Path

from audiodata import MultitaskDataset
from audiodata.datasets import collate_fn
from audiodata.utils import get_captioning_data
from audiodata.utils import get_scenes_data
from audiodata.utils import get_events_data

from common import init_log
from common import read_yaml

from models import Transformer
from models import ModelParameters


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


def masked_loss(y_pred, y_true, mask):
    loss = F.cross_entropy(y_pred.permute(0, 2, 1), y_true, reduction='none')
    mask = (mask == 0).to(loss.dtype)
    return torch.sum(loss * mask) / torch.sum(mask).to(loss.dtype)


def step_lr(step, d_model, warmup_steps=4000):
    arg1 = torch.tensor(1 / math.sqrt(step)) if step > 0 else torch.tensor(float('inf'))
    arg2 = torch.tensor(step * warmup_steps**-1.5)
    
    return 1 / math.sqrt(d_model) * torch.minimum(arg1, arg2)


def model_size(model):
    num_params = sum(param.numel() for param in model.parameters())
    return num_params


def main(config_fn='settings.yaml'):
    cfg = read_yaml(config_fn)
    log_level = cfg.get('log_level', 'info')
    logger = init_log('train', level=log_level)

    logger.info('reading data')
    caption_data = get_captioning_data(cfg)
    logger.info(f'got {len(caption_data)} audio - caption pairs')

    scene_data = get_scenes_data(cfg)
    logger.info(f'got {len(scene_data)} audio - scene label pairs')

    event_data = get_events_data(cfg)
    logger.info(f'got {len(event_data)} audio - event list pairs')

    sample_rate = cfg.get('sample_rate', 16000)
    n_fft = cfg.get('n_fft', 1024)
    hop_length = cfg.get('hop_length', 512)
    n_mels = cfg.get('n_mels', 128)

    logger.info(f'sample_rate={sample_rate}, n_fft={n_fft}, hop_length={hop_length}, n_mels={n_mels}')

    # use default tokenizer
    train_dataset = MultitaskDataset(
        caption_data=caption_data,
        scene_data=scene_data,
        event_data=event_data,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels)

    n_tokens = train_dataset.vocab_size()

    batch_size = cfg.get('batch_size', 8)
    epochs = cfg.get('epochs', 10)
    num_workers = cfg.get('num_dataloader_workers', 4)

    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    params = ModelParameters(
        d_model=cfg.get('d_model', 128),
        d_ff=cfg.get('d_ff', 512),
        n_enc_heads=cfg.get('n_enc_heads', 4),
        n_enc_layers=cfg.get('n_enc_layers', 2),
        n_dec_heads=cfg.get('n_dec_heads', 4),
        n_dec_layers=cfg.get('n_dec_layers', 2),
        n_mels=n_mels,
        n_tokens=n_tokens,
        max_mel_length=cfg.get('max_mel_length', 512),
        max_sequence_length=cfg.get('max_sequence_length', 64),
        dropout=cfg.get('dropout'))

    
    transformer = Transformer(params).to(device)
    logger.info(f'transformer size {model_size(transformer)/1e6:.1f}M')

    optimizer = torch.optim.Adam(transformer.parameters(), lr=1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step_lr(step, params.d_model, warmup_steps=6000))
    log_interval = cfg.get('log_interval', 100)
    model_path = Path(cfg.get('model_path', 'model'))
    model_path.mkdir(exist_ok=True, parents=True)

    for epoch in range(epochs):
        t0 = time.time()
        batch_t0 = t0
        
        for batch, (mels, mel_mask, tokens, token_mask) in enumerate(train_loader):
            mels = mels.to(device)
            mel_mask = mel_mask.to(device)
            tokens = tokens.to(device)
            token_mask = token_mask.to(device)

            # split tokens
            inp_tokens = tokens[:, :-1]
            token_mask = token_mask[:, :-1]

            tar_tokens = tokens[:, 1:]

            y_pred = transformer(mels, inp_tokens, mel_mask, token_mask)

            loss = masked_loss(y_pred, tar_tokens, token_mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if batch % log_interval == 0:
                t_batch = (time.time() - batch_t0) * 1000 / log_interval

                current_lr = optimizer.param_groups[0]['lr']
                print(f'batch {batch:5d}/{len(train_loader)} | {int(t_batch):5d} ms/batch | learning rate {current_lr:.4g} | train loss {loss.item():.4f}')
                batch_t0 = time.time()

        # save every epoch:
        torch.save(transformer.state_dict(), model_path / f'model-{epoch + 1:02d}.pt')


if __name__ == '__main__':
    main()
