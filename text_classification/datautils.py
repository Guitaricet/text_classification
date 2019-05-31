import re

import numpy as np
import pandas as pd

import torch
import torchtext

from gensim.models import FastText
from pymystem3 import Mystem

import cfg
from text_classification.utils import noise_generator
# TODO: move textlen params from class to object properties


class HierarchicalIMDB(torchtext.datasets.imdb.IMDB):
    """
    Dataset class for hierarchical (chars -> words -> text) networks, IMDB dataset

    Zero vector used for padding
    """
    noise_level = 0
    alphabet = cfg.alphabet
    max_text_len = cfg.max_text_len
    max_word_len = cfg.max_word_len

    def __init__(self, alphabet=None, **kwargs):
        self.alphabet = alphabet or self.alphabet
        self.char2int = {s: i for i, s in enumerate(self.alphabet)}
        self.unk_index = self.char2int['<UNK>']
        self.pad_index = self.char2int['<PAD>']
        super().__init__(**kwargs)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        _text_tensor = self.preprocess(item.text)

        label = int(item.label == 'pos')
        return _text_tensor, label

    def preprocess(self, text, with_noise=True):
        _text_tensor = torch.zeros([self.max_text_len, self.max_word_len])

        for i, token in enumerate(text):
            if i >= self.max_text_len: break  # noqa: E701
            for j, char in enumerate(token):
                if j >= self.max_word_len: break  # noqa: E701
                _text_tensor[i, j] = self.char2int.get(char, self.unk_index)
        return _text_tensor

    def _noise_generator(self, string):
        return noise_generator(string, self.noise_level, self.alphabet)


class HierarchicalCSVDataset(torch.utils.data.Dataset):
    """
    Dataset class for hierarchical (chars -> words -> text) networks which reads data from .csv

    Mokoron, because it was firstly used for Mokoron twitter sentiment dataset
    Zero vector used for padding
    """
    noise_level = 0
    alphabet = cfg.alphabet

    # TODO: rename maxwordlen and maxtextlen
    def __init__(self,
                 filepath,
                 text_field,
                 label_field,
                 max_word_len=cfg.max_word_len,
                 max_text_len=cfg.max_text_len,
                 alphabet=None):

        self.data = pd.read_csv(filepath)
        self.alphabet = alphabet or self.alphabet
        self.mystem = Mystem()
        self.text_field = text_field
        self.label_field = label_field
        self.max_word_len = max_word_len
        self.max_text_len = max_text_len
        self.char2int = {s: i for i, s in enumerate(self.alphabet)}
        self.unk_index = self.char2int['<UNK>']
        self.pad_index = self.char2int['<PAD>']
        self.label2int = {l: i for i, l in enumerate(sorted(self.data[self.label_field].unique()))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # processing from char representation is required for text noising
        line = self.data.iloc[idx]
        text = line[self.text_field].lower()
        label = self.label2int[line[self.label_field]]

        if self.noise_level > 0:
            text = self._noise_generator(text)

        text = self._tokenize(text)
        text = self._numericalize(text)
        return text, label

    def _noise_generator(self, string):
        return noise_generator(string, self.noise_level, self.alphabet)

    def _tokenize(self, text):
        return [res['text'] for res in self.mystem.analyze(text) if res['text'] != ' ']

    def _numericalize(self, text):
        _text_tensor = torch.zeros([self.max_text_len, self.max_word_len])

        for i, token in enumerate(text):
            if i >= self.max_text_len: break  # noqa: E701
            for j, char in enumerate(token):
                if j >= self.max_word_len: break  # noqa: E701
                _text_tensor[i, j] = self.char2int.get(char, self.unk_index)

        return _text_tensor

    def onehot2text(self, one_hotted_text, batch_size=None, show_pad=False):
        if batch_size is None:
            text = ''
            max_values, idx = torch.max(one_hotted_text, 1)
            for c, i in enumerate(idx):
                if max_values[c] == 0:
                    if show_pad:
                        symb = '<PAD>'
                    else:
                        symb = ''
                else:
                    symb = cfg.alphabet[i]
                text += symb
            return text
        else:
            texts = [self.onehot2text(oht, batch_size=batch_size) for oht in one_hotted_text]
            return texts


class FastTextIMDB(torchtext.datasets.imdb.IMDB):
    noise_level = 0
    alphabet = cfg.alphabet
    embeddings = None

    def __init__(self,
                 path,
                 text_field,
                 label_field,
                 embeddings,
                 max_text_len=cfg.max_text_len,
                 **kwargs):
        """
        IMDB with FastText embeddings dataset

        Zero vector used for padding
        """
        super().__init__(path, text_field, label_field, **kwargs)
        assert isinstance(embeddings, [str, FastText]),\
            'embeddings should be gensim FastText object or path to FastText file'
        self.embeddings = embeddings

        if isinstance(embeddings, str):
            print('Loading embeddings from file')
            self.embeddings = FastText.load_fasttext_format(embeddings)

        self.max_text_len = max_text_len
        self.unk_vec = np.random.rand(self.embeddings.vector_size)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        # ???
        # indices ordered differently from previous models — (word_vec, word_num) instead of (word_num, word_vec)
        _text_tensor = torch.zeros([self.max_text_len, self.embeddings.vector_size])

        for i, token in enumerate(item.text):
            if i >= self.max_text_len: break  # noqa: E701

            token = self._noise_generator(token)
            token_vec = self.embeddings.get(token, self.unk_vec)

            token_tensor = torch.FloatTensor(token_vec)
            _text_tensor[i, :] = token_tensor

        label = int(item.label == 'pos')
        return _text_tensor, label

    def _noise_generator(self, string):
        return noise_generator(string, self.noise_level, self.alphabet)


class FastTextCSVDataset(torch.utils.data.Dataset):
    """
    Zero vector used for padding
    """
    noise_level = 0

    def __init__(self,
                 filepath,
                 text_field,
                 label_field,
                 embeddings=None,
                 max_text_len=cfg.max_text_len,
                 alphabet=None,
                 elmo=False):
        if isinstance(embeddings, str):
            self.embeddings = FastText.load_fasttext_format(embeddings)
        elif isinstance(embeddings, FastText):
            self.embeddings = embeddings
        elif embeddings is None:
            self.embeddings = None
            assert elmo
        else:
            raise ValueError('embeddings should be path to FastText file of gensim FastText object or None')

        self.alphabet = alphabet or cfg.alphabet
        self.elmo = elmo
        self.mystem = Mystem()
        self.text_field = text_field
        self.label_field = label_field
        self.data = pd.read_csv(filepath)
        self.max_text_len = max_text_len
        if self.embeddings is not None:
            self.unk_vec = np.random.rand(self.embeddings.vector_size)
        self.label2int = {l: i for i, l in enumerate(sorted(self.data[self.label_field].unique()))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data.iloc[idx]
        text = line[self.text_field].lower()
        # for mokoron dataset we should remove smiles
        if ')' not in self.alphabet:
            text = re.sub('[(,)]', '', text)
        label = self.label2int[line[self.label_field]]

        if self.noise_level > 0:
            text = self._noise_generator(text)

        text = self._tokenize(text)
        if not self.elmo:
            text = self._preprocess(text)
        return text, label

    def _tokenize(self, text):
        return [res['text'] for res in self.mystem.analyze(text) if res['text'] != ' ']

    def _preprocess(self, text):
        # indicies orded different from previous models — (word_vec, word_num) instead of (word_num, word_vec)
        _text_tensor = torch.zeros([self.max_text_len, self.embeddings.vector_size])

        for i, token in enumerate(text):
            if i >= self.max_text_len: break  # noqa: E701

            token = self._noise_generator(token)
            token_vec = self.embeddings.get(token, self.unk_vec)

            token_tensor = torch.FloatTensor(token_vec)
            _text_tensor[i, :] = token_tensor

        return _text_tensor

    def _noise_generator(self, string):
        return noise_generator(string, self.noise_level, self.alphabet)


# --- Functions


def model_params_num(model):
    return sum(np.prod(list(p.size())) for p in model.parameters())


def mk_dataline(model_type, epochs, lr, noise_level_train, noise_level_test, acc_train, acc_test,
                f1_train, f1_test, dropout, model, run_name,):
    # TODO: do not use cfg here?
    return {
        'model_type': model_type,
        'trainable_params': model_params_num(model),
        'dropout': dropout,
        'epochs': epochs,
        'lr': lr,
        'noise_level_train': noise_level_train,
        'noise_level_test': noise_level_test,
        'acc_train': acc_train,
        'acc_test': acc_test,
        'f1_train': f1_train,
        'f1_test': f1_test,
        'model_desc': str(model),
        'run_name': run_name,
        'data_desc': 'MaxWordLen %s, MaxTexLen %s' % (cfg.max_word_len, cfg.max_text_len)
    }
