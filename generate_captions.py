

from pathlib import Path
from tqdm import tqdm

import pandas as pd

import torch
import torchaudio
from transformers import BertTokenizer

from common import read_yaml
from common import read_pickle
from models import Transformer
from models.utils import load_transformer


def generate_caption(model, mels, tokenizer, max_length=128):
    start_token, end_token = tokenizer.encode('')

    tokens = torch.tensor([start_token], dtype=torch.int32)
    
    mel_mask = torch.zeros(1, mels.size(1))

    new_token = 0
    k = 1
    while new_token != end_token:
        if k < max_length:
            token_mask = torch.zeros(1, k)
            preds = model(mels.unsqueeze(0), tokens.unsqueeze(0), mel_mask, token_mask)
    
            new_token = torch.argmax(preds[0, -1:, :])
        else:
            new_token = end_token

        tokens = torch.cat([tokens, new_token.unsqueeze(0)])
        k = tokens.size(0)

    return tokenizer.decode(tokens[1:-1])


def predict_captions(audio_fns, model, cfg):
    captions = {}

    sample_rate = cfg.get('sample_rate', 16000)
    n_fft = cfg.get('n_fft', 1024)
    hop_length = cfg.get('hop_length', 512)
    n_mels = cfg.get('n_mels', 128)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    for fn in tqdm(audio_fns):
        y, sr = torchaudio.load(fn)
        y = torch.mean(y, axis=0)
        if sr != sample_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=sample_rate)

        mels = mel_spec(y)
        
        captions[Path(fn).name] = generate_caption(model, mels, tokenizer)

    return captions


def main(config_fn='eval_settings.yaml'):
    eval_cfg = read_yaml(config_fn)

    # train cfg
    cfg_fn = eval_cfg.get('config_fn', 'settings.yaml')
    train_cfg = read_yaml(cfg_fn)
    
    params = read_pickle('train_params.pkl')
    transformer = load_transformer('data/transformer/model-10.pt', params)
    transformer.eval()

    caption_cfg = eval_cfg.get('captions')

    if caption_cfg is None:
        print('nothing in cfg')
        return

    with torch.no_grad():

        if 'clotho_dir' in caption_cfg:
            clotho_dir = Path(caption_cfg['clotho_dir'])
            csv_fn = clotho_dir / 'csv' / 'clotho_captions_evaluation.csv'
            df = pd.read_csv(csv_fn, index_col=0, encoding='latin1')
            filenames = list(map(lambda fn: clotho_dir / 'audio' / 'evaluation' / fn, df.index))

            captions = predict_captions(filenames, transformer, train_cfg)

            df = pd.DataFrame({'caption': pd.Series(captions)})
            df.index.name = 'file_name'
            df.to_csv('predictions_clotho.csv')

if __name__ == '__main__':
    main()
