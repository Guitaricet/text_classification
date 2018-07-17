from random import random, choice

import numpy as np
import pandas as pd

import torch
import torchtext

from gensim.models import FastText
from pymystem3 import Mystem


import cfg

# TODO: move textlen params from class to object properties
# TODO: move char2int to classes
# NOTE: move noise generator somewhere?
char2int = {s: i for s, i in zip(cfg.alphabet, range(len(cfg.alphabet)))}


class HierarchicalIMDB(torchtext.datasets.imdb.IMDB):
    """
    Dataset class for hierarchical (chars -> words -> text) networks, IMDB dataset

    Zero vector used for padding
    """
    noise_level = 0
    alphabet = cfg.alphabet
    max_text_len = cfg.max_text_len
    max_word_len = cfg.max_word_len

    def __getitem__(self, idx):
        item = super(HierarchicalIMDB, self).__getitem__(idx)
        _text_tensor = self.preprocess(item.text)

        label = int(item.label == 'pos')
        return _text_tensor, label
    
    def preprocess(self, text, with_noise=True):
        _text_tensor = torch.zeros([self.max_word_len * self.max_text_len, len(self.alphabet)])

        for i, token in enumerate(text):
            if i >= self.max_text_len:
                break
            if with_noise:
                token = self._noise_generator(token)
            for j, char in enumerate(token):
                if j >= self.max_word_len:
                    break
                _text_tensor[i*self.max_word_len + j, char2int.get(char, char2int['<UNK>'])] = 1.
        return _text_tensor

    def _noise_generator(self, string):
        noised = ""
        for c in string:
            if random() > self.noise_level:
                noised += c
            if random() < self.noise_level:
                noised += choice(self.alphabet)
        return noised


class HierarchicalMokoron(torch.utils.data.Dataset):
    """
    Dataset class for hierarchical (chars -> words -> text) networks which reads data from .csv

    Mokoron, because it was firstly used for Mokoron twitter sentiment dataset 
    Zero vector used for padding
    """
    noise_level = 0
    alphabet = cfg.alphabet
    max_text_len = cfg.max_text_len
    max_word_len = cfg.max_word_len

    # TODO: rename maxwordlen and maxtextlen
    def __init__(self,
                 filepath,
                 text_field,
                 label_field,
                 maxwordlen=cfg.max_word_len,
                 maxtextlen=cfg.max_text_len,
                 alphabet=None):

        self.alphabet = alphabet or cfg.alphabet
        self.mystem = Mystem()
        self.text_field = text_field
        self.label_field = label_field
        self.data = pd.read_csv(filepath)
        self.maxwordlen = maxwordlen
        self.maxtextlen = maxtextlen
        self.char2int = {s: i for s, i in zip(self.alphabet, range(len(self.alphabet)))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data.iloc[idx]
        text = line[self.text_field].lower()
        label = int(line[self.label_field] == 1.)

        if self.noise_level > 0:
            text = self._noise_generator(text)

        text = self._tokenize(text)
        text = self._preprocess(text)
        return text, label

    def _tokenize(self, text):
        return [res['text'] for res in self.mystem.analyze(text) if res['text'] != ' ']

    def _noise_generator(self, string):
        noised = ''
        for c in string:
            if random() > self.noise_level:
                noised += c
            if random() < self.noise_level:
                noised += choice(self.alphabet)
        return noised

    def _one_hot(self, char):
        zeros = np.zeros(len(self.alphabet))
        if char in self.char2int:
            zeros[self.char2int[char]] = 1.
        else:
            zeros[self.char2int['<UNK>']] = 1.

    def _preprocess(self, text):
        _text_tensor = torch.zeros([self.maxwordlen * self.maxtextlen, len(self.alphabet)])
        
        for i, token in enumerate(text):
            if i >= self.maxtextlen:
                break
            for j, char in enumerate(token):
                if j >= self.maxwordlen:
                    break
                _text_tensor[i*self.maxwordlen + j, self.char2int.get(char, self.char2int['<UNK>'])] = 1.

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
            texts = [self.onehot2text(oht, batch_size=None) for oht in one_hotted_text]
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
        super(FastTextIMDB, self).__init__(path, text_field, label_field, **kwargs)
        if isinstance(embeddings, str):
            print('Loading embeddings from file')
            self.embeddings = FastText.load_fasttext_format(embeddings)
        elif isinstance(embeddings, FastText):
            print('Got embeddings')
            self.embeddings = embeddings
        else:
            raise ValueError('embeddings should be path to FastText file of gensim FastText object')
        self.max_text_len = max_text_len
        self.unk_vec = np.random.rand(self.embeddings.vector_size)

    def __getitem__(self, idx):
        item = super(FastTextIMDB, self).__getitem__(idx)
        # indices ordered differently from previous models — (word_vec, word_num) instead of (word_num, word_vec)
        _text_tensor = torch.zeros([self.max_text_len, self.embeddings.vector_size])

        for i, token in enumerate(item.text):
            if i >= self.max_text_len:
                break

            token = self._noise_generator(token)
            if token in self.embeddings:
                token_vec = self.embeddings[token]
            else:
                token_vec = self.unk_vec

            token_tensor = torch.FloatTensor(token_vec)
            _text_tensor[i, :] = token_tensor

        label = int(item.label == 'pos')
        return _text_tensor, label

    def _noise_generator(self, string):
        noised = ""
        for c in string:
            if random() > self.noise_level:
                noised += c
            if random() < self.noise_level:
                noised += choice(self.alphabet)
        return noised


class FastTextMokoron(torch.utils.data.Dataset):
    """
    Zero vector used for padding
    """
    noise_level = 0

    def __init__(self, filepath, text_field, label_field, embeddings, maxlen=cfg.max_text_len, alphabet=None):
        if isinstance(embeddings, str):
            self.embeddings = FastText.load_fasttext_format(embeddings)
        elif isinstance(embeddings, FastText):
            self.embeddings = embeddings
        else:
            raise ValueError('embeddings should be path to FastText file of gensim FastText object')

        self.alphabet = alphabet or cfg.alphabet
        self.mystem = Mystem()
        self.text_field = text_field
        self.label_field = label_field
        self.data = pd.read_csv(filepath)
        self.max_text_len = maxlen
        self.unk_vec = np.random.rand(self.embeddings.vector_size)

    def __len__(self):
        return len(self.data)

    def _tokenize(self, text):
        return [res['text'] for res in self.mystem.analyze(text) if res['text'] != ' ']

    def __getitem__(self, idx):
        line = self.data.iloc[idx]
        text = line[self.text_field].lower()
        # for mokoron dataset we should remove smiles
        if ')' not in self.alphabet:
            text = [t for t in text if t not in ('(', ')')]
        label = int(line[self.label_field] == 1.)

        if self.noise_level > 0:
            text = self._noise_generator(text)

        text = self._tokenize(text)
        text = self._preprocess(text)
        return text, label

    def _preprocess(self, text):
        # indicies orded different from previous models — (word_vec, word_num) instead of (word_num, word_vec)
        _text_tensor = torch.zeros([self.max_text_len, self.embeddings.vector_size])

        for i, token in enumerate(text):
            if i >= self.max_text_len:
                break

            if token in self.embeddings:
                token_vec = self.embeddings[token]
            else:
                token_vec = self.unk_vec

            token_tensor = torch.FloatTensor(token_vec)
            _text_tensor[i, :] = token_tensor

        return _text_tensor

    def _noise_generator(self, string):
        noised = ""
        for c in string:
            if random() > self.noise_level:
                noised += c
            if random() < self.noise_level:
                noised += choice(self.alphabet)
        return noised


class CharIMDB(torchtext.datasets.imdb.IMDB):
    noise_level = 0
    alphabet = cfg.alphabet
    maxlen = cfg.max_text_len * cfg.max_word_len

    def __getitem__(self, idx):
        item = super(CharIMDB, self).__getitem__(idx)
        text = item.text
        text = self._noise_generator(text)
        label = int(item.label == 'pos')
        return self._preprocess(text), label

    def _preprocess(self, text):
        one_hotted_text = np.zeros((self.maxlen, len(self.alphabet)))
        for i, char in enumerate(text):
            if i >= self.maxlen:
                break
            one_hotted_text[i, char2int.get(char, char2int['<UNK>'])] = 1.

        return torch.FloatTensor(one_hotted_text)

    def _noise_generator(self, string):
        noised = ""
        for c in string:
            if random() > self.noise_level:
                noised += c
            if random() < self.noise_level:
                noised += choice(self.alphabet)
        return noised


class CharMokoron(torch.utils.data.Dataset):
    """
    Zero vector for padding.
    """
    noise_level = 0
    maxlen = cfg.max_text_len * cfg.max_word_len

    def __init__(self, filepath, text_field, label_field, maxlen=None, alphabet=None):

        self.data = pd.read_csv(filepath)
        self.text_field = text_field
        self.label_field = label_field
        if maxlen is not None:
            self.maxlen = maxlen
        self.alphabet = alphabet or cfg.alphabet
        self.char2int = {s: i for s, i in zip(self.alphabet, range(len(self.alphabet)))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data.iloc[idx]
        text = line[self.text_field]
        label = int(line[self.label_field] == 1.)

        if self.noise_level > 0:
            text = self._noise_generator(text)
        text = self._preprocess(text)
        return text, label

    def _noise_generator(self, string):
        noised = ""
        for c in string:
            if random() > self.noise_level:
                noised += c
            if random() < self.noise_level:
                noised += choice(self.alphabet)
        return noised

    def _one_hot(self, char):
        zeros = np.zeros(len(self.alphabet))
        if char in self.char2int:
            zeros[self.char2int[char]] = 1.
        else:
            zeros[self.char2int['<UNK>']] = 1.

    def _preprocess(self, text):
        text = text.lower()
        one_hotted_text = np.zeros((self.maxlen, len(self.alphabet)))
        for i, char in enumerate(text):
            if i >= self.maxlen:
                break
            one_hotted_text[i, self.char2int.get(char, self.char2int['<UNK>'])] = 1.

        return torch.FloatTensor(one_hotted_text)

    def onehot2text(self, one_hotted_text, show_pad=False):
        text = ''
        max_values, idx = torch.max(one_hotted_text, 1)
        for c, i in enumerate(idx):
            if max_values[c] == 0:
                if show_pad:
                    symb = '<PAD>'
                else:
                    symb = ''
            else:
                symb = self.alphabet[i]
            text += symb
        return text

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
