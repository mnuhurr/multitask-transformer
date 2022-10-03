
import audiofile as af
import pandas as pd

from pathlib import Path
from operator import itemgetter
from tqdm import tqdm


# captions
def get_clotho_data(clotho_dir, split='development'):
    clotho_dir = Path(clotho_dir)

    df = pd.read_csv(clotho_dir / 'csv' / f'clotho_captions_{split}.csv', index_col=0, encoding='latin1')
    
    data = []

    for fn, captions in df.iterrows():
        audio_fn = clotho_dir / 'audio' / split / fn

        if not audio_fn.exists():
            continue

        for caption in captions:
            if caption is not None:
                data.append((audio_fn, caption))

    return data


def get_audiocaps_data(audiocaps_dir, split='train', min_duration=None):
    audiocaps_dir = Path(audiocaps_dir)
    df = pd.read_csv(audiocaps_dir / 'dataset' / f'{split}.csv', index_col=0)

    data = []

    for fn, row in df.iterrows():
        audio_fn = audiocaps_dir / 'audio' / split / f'{fn}.wav'

        if not audio_fn.exists():
            continue

        if min_duration is not None and af.duration(audio_fn) < min_duration:
            continue

        data.append((audio_fn, row['caption']))

    return data


def get_captioning_data(cfg, split='development'):
    
    data = []
    if split == 'development':
        ac_split = 'train'
    elif split == 'validation':
        ac_split = 'val'
    elif split == 'evaluation':
        ac_split = 'test'

    if 'clotho_dir' in cfg:
        data.extend(get_clotho_data(cfg['clotho_dir'], split=split))

    if 'audiocaps_dir' in cfg:
        data.extend(get_audiocaps_data(cfg['audiocaps_dir'], split=ac_split))

    return data


# scenes
def get_scenes_dir_data(data_dir):
    data_dir = Path(data_dir)

    if (data_dir / 'meta.txt').exists():
        df = pd.read_csv(data_dir / 'meta.txt', delimiter='\t', header=None, index_col=0)

    elif (data_dir / 'meta.csv').exists():
        df = pd.read_csv(data_dir / 'meta.csv', delimiter='\t', index_col=0)

    data = []
    col = df.columns[0]

    for idx, row in df.iterrows():
        data.append((data_dir / idx, row[col]))

    return data


def get_scenes_data(cfg):
    data = []

    data_dirs = cfg.get('scenes_datasets', [])
    if data_dirs is None:
        return data

    for data_dir in data_dirs:
        data.extend(get_scenes_dir_data(data_dir))

    return data


# events
def get_audioset_data(fn, audio_dir, min_duration=None):
    # every element of the return array is a (file, event) -pair
    df = pd.read_csv(fn, delimiter='\t', index_col=0)

    audio_dir = Path(audio_dir)

    data = []

    for idx, row in df.iterrows():
        audio_fn = audio_dir / f'{idx}.wav'
        if audio_fn.exists():
            if min_duration is not None:
                if af.duration(audio_fn) < min_duration:
                    continue

            if len(row) == 1 or len(row) == 3:
                data.append([audio_fn] + list(row))

    return data

def get_audioset_data_combined(fn, audio_dir, min_duration=None):
    # return list contains (file, list of events)
    df = pd.read_csv(fn, delimiter='\t', index_col=0)

    audio_dir = Path(audio_dir)

    data = {}

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_fn = audio_dir / f'{idx}.wav'

        if audio_fn in data:
            # we have already checked that the audio is okay
            data[audio_fn].append(list(row))
        elif audio_fn.exists():
            if min_duration is not None:
                if af.duration(audio_fn) < min_duration:
                    continue

            data[audio_fn] = [list(row)]

    return list(zip(data.keys(), data.values()))

def get_events_data(cfg):
    data = []

    if 'audioset_csv' in cfg and 'audioset_data_dir' in cfg:
        min_duration = cfg.get('min_duration')
        data.extend(get_audioset_data_combined(cfg['audioset_csv'], cfg['audioset_data_dir'], min_duration=min_duration))

    return data
