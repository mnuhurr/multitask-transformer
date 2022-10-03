
import torch
from transformers import BertTokenizer


class Tokenizer:
    def __init__(self, scene_labels, event_labels, text_tokenizer=None):

        self.tokenizer = text_tokenizer if text_tokenizer is not None else BertTokenizer.from_pretrained('bert-base-uncased')

