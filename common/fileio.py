
from pathlib import Path
import json
import yaml
import pickle

def write_list(fn, word_list):
    with Path(fn).open('wt') as f:
        for word in word_list:
            print(word, file=f)


def read_yaml(filename):
    with Path(filename).open('rt', encoding='utf8') as file:
        data = yaml.safe_load(file)
        return data


def read_json(filename):
    with Path(filename).open('rt', encoding='utf8') as file:
        data = json.load(file)
        return data


def write_json(filename, data):
    with Path(filename).open('wt', encoding='utf8') as file:
        json.dump(data, file)
    

def read_pickle(filename):
    with Path(filename).open('rb') as file:
        data = pickle.load(file)
        return data


def write_pickle(filename, data):
    with Path(filename).open('wb') as file:
        pickle.dump(data, file)
