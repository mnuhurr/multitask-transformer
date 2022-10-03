
import torch
import torchaudio
from transformers import BertTokenizer
from operator import itemgetter


"""
token range:
    - tokenizer
    - special tokens
    - scene classes
    - 
"""

class MultitaskDataset(torch.utils.data.Dataset):
    def __init__(self, caption_data=None, scene_data=None, event_data=None, tokenizer=None, n_fft=1024, hop_length=None, n_mels=128, sample_rate=16000, max_mel_length=768):
        """
        all data: lists of (filename, output) pairs


        """
        if hop_length is None:
            hop_length = n_fft // 2

        self.tokenizer = tokenizer if tokenizer is not None else BertTokenizer.from_pretrained('bert-base-uncased')
        self.end_token = self.tokenizer.encode('')[-1]

        self.caption_data = caption_data if caption_data is not None else []
        self.scene_data = scene_data if scene_data is not None else []
        self.event_data = event_data if event_data is not None else []

        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        self.mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

        # find different classes
        self.scene = self.get_scene_labels(self.scene_data)
        self.scene_index = {self.scene[k]: k for k in range(len(self.scene))}

        # use same acoustic event tokens for both strong & weak annotation. (todo: change?)
        self.event = self.get_event_labels(self.event_data)
        self.event_index = {self.event[k]: k for k in range(len(self.event))}

        self.special_tokens = {
            'start_scene_cls': len(self.tokenizer),
            'start_event_cls': len(self.tokenizer) + 1,
            'start_event_time_cls': len(self.tokenizer) + 2,
        }
        
        self.max_mel_length = max_mel_length


    def get_scene_labels(self, data):
        # data = [(fn_1, label_1), (fn_2, label_2), ...]
        labels = map(itemgetter(1), data)
        return sorted(set(labels))

    def get_event_labels(self, data):
        # data = [(fn_1, [event_1, event_2, ...]), (fn_2, [event_n, ...])]
        event_lists = map(itemgetter(1), data)
        labels = set()
        for event_list in event_lists:
            for event in event_list:
                labels.add(event[0])

        return sorted(set(labels))

    def vocab_size(self):
        # use different tokens for onset/offset
        num_timestep_tokens = 2 * self.max_mel_length
        return len(self.tokenizer) + len(self.scene) + len(self.event) + len(self.special_tokens) + num_timestep_tokens

    def __len__(self):
        return len(self.caption_data) + len(self.scene_data) + len(self.event_data)

    def get_mels(self, filename):
        y, sr = torchaudio.load(filename)
        
        # make sure to be mono
        y = torch.mean(y, axis=0)

        if sr != self.sample_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sample_rate)

        # shape is (n_mels, t)
        mels = self.mel_spec(y)
        offset = 0
        
        if mels.size(1) > self.max_mel_length:
            offset = torch.randint(mels.size(1) - self.max_mel_length, size=(1,))
            mels = mels[:, offset:offset + self.max_mel_length]

        assert mels.size(0) == self.n_mels
        assert mels.size(1) <= self.max_mel_length

        return mels, offset

    def get_scene_class_token(self, scene_class):
        offset = len(self.tokenizer) + len(self.special_tokens)
        idx = self.scene_index[scene_class]

        return offset + idx

    def get_event_class_token(self, event_class):
        offset = len(self.tokenizer) + len(self.special_tokens) + len(self.scene_index)
        idx = self.event_index[event_class]

        return offset + idx

    def get_event_time_tokens(self, start_time, end_time):
        offset = len(self.tokenizer) + len(self.special_tokens) + len(self.scene_index) + len(self.event_index)

        f_0 = int(start_time * self.sample_rate) // self.hop_length
        f_1 = int(end_time * self.sample_rate) // self.hop_length

        t_0 = offset + f_0
        t_1 = offset + self.max_mel_length + f_1

        return t_0, t_1

    def get_caption_item(self, item):
        # item is relative to the data subset
        fn, output = self.caption_data[item]
        mels, _ = self.get_mels(fn)

        return mels, self.tokenizer.encode(output)

    def get_scene_item(self, item):
        # item is relative to the data subset
        fn, scene_cls = self.scene_data[item]
        mels, _ = self.get_mels(fn)

        cls_token = self.get_scene_class_token(scene_cls)

        tokens = [self.special_tokens['start_scene_cls'], cls_token, self.end_token]

        return mels, tokens

    def get_event_item(self, item):
        # item is relative to the data subset
        fn, event_list = self.event_data[item]

        mels, mel_offset = self.get_mels(fn)

        tokens = []
        

        if len(event_list[0]) == 1:
            # weak annotation: event class only
            tokens.append(self.special_tokens['start_event_cls'])
            
            for event_data in event_list:
                cls_token = self.get_event_class_token(event_data[0])
                tokens.append(cls_token)

        elif len(event_list[0]) == 3:
            # strong annotation: event class + time
            tokens.append(self.special_tokens['start_event_time_cls'])

            for event_data in event_list:
                cls_token = self.get_event_class_token(event_data[0])
                tokens.append(cls_token)
        
                # for strong annotation use time index for corresponding mel frame. i.e. 
                # f = t * sample_rate / hop_length,
                start_time, end_time = event_data[1:]

                t_0, t_1 = self.get_event_time_tokens(start_time, end_time)
            
                t_0 = t_0 - mel_offset
                t_1 = min(t_1, mels.size(1) - 1)

                tokens.append(t_0)
                tokens.append(t_1)
        else:
            raise ValueError('invalid event data:', event_data)
            
        tokens.append(self.end_token)

        return mels, tokens

    def __getitem__(self, item):
        if item < len(self.caption_data):
            mels, tokens = self.get_caption_item(item)

        elif item < len(self.caption_data) + len(self.scene_data):
            mels, tokens = self.get_scene_item(item - len(self.caption_data))

        elif item < len(self.caption_data) + len(self.scene_data) + len(self.event_data):
            mels, tokens = self.get_event_item(item - len(self.caption_data) - len(self.scene_data))

        tokens = torch.tensor(tokens, dtype=torch.long)

        return mels, tokens
    

def collate_fn(batch):
    batch_size = len(batch)
    audio_lens = list(map(lambda t: t.size(1), map(itemgetter(0), batch)))
    seq_lens = list(map(len, map(itemgetter(1), batch)))

    max_audio_len = max(audio_lens)
    max_seq_len = max(seq_lens)

    audio_dim = batch[0][0].size(0)

    audio = torch.zeros(batch_size, audio_dim, max_audio_len, dtype=batch[0][0].dtype)
    seq = torch.zeros(batch_size, max_seq_len, dtype=batch[0][1].dtype)
    audio_mask = -float('inf') * torch.ones(batch_size, max_audio_len)
    seq_mask = -float('inf') * torch.ones(batch_size, max_seq_len)

    for k, (mels, tokens) in enumerate(batch):
        audio_len = mels.size(1)
        audio[k, :, :audio_len] = mels
        audio_mask[k, :audio_len] = 0.0

        seq_len = tokens.size(0)
        seq[k, :seq_len] = tokens
        seq_mask[k, :seq_len] = 0.0

    return audio, audio_mask, seq, seq_mask




def foo():
    #caption_data = [('/home/work/mnu/data/clotho/audio/development/zipper.wav', 'zipper what')]
    #scene_data = [('/home/work/mnu/data/TAU-urban-acoustic-scenes-2019-development/audio/metro_station-helsinki-231-6950-a.wav', 'metro-station'),
    #        ('/home/work/mnu/data/TAU-urban-acoustic-scenes-2019-development/audio/park-barcelona-90-2483-a.wav', 'park')]

    #        ('/home/work/mnu/data/TAU-urban-acoustic-scenes-2019-development/audio/park-barcelona-90-2483-a.wav', 'park', 4.0, 7.0)]

    caption_data = None
    scene_data = None

    from utils import get_audioset_data_combined
    print('loading data')
    event_data = get_audioset_data_combined('/home/work/mnu/src/GoogleAudioSetReformatted/audioset_strong_train.tsv', 
            '/home/work/mnu/data/audioset_strong', min_duration=7.0) 
    print('done loading')

    mtd = MultitaskDataset(caption_data=caption_data, scene_data=scene_data, event_data=event_data)

    it = iter(mtd)
    for k in range(len(mtd)):
        m, t = next(it)
        print(m.shape, t)
        break

    print(mtd.vocab_size(), len(mtd.tokenizer))

if __name__ == '__main__':
    foo()
