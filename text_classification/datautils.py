import re

import numpy as np
import pandas as pd

import torch
import torchtext

from nltk.tokenize import word_tokenize
from gensim.models.keyedvectors import FastTextKeyedVectors, KeyedVectors
from gensim.models.fasttext import load_facebook_vectors


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
        _text_len = min(self.max_text_len, len(text))
        _text_tensor = torch.zeros([_text_len, self.max_word_len])

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
        return word_tokenize(text)

    def _numericalize(self, text):
        _text_len = min(self.max_text_len, len(text))
        _text_tensor = torch.zeros([_text_len, self.max_word_len], dtype=torch.long)

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
        if not isinstance(embeddings, (str, FastTextKeyedVectors, KeyedVectors)):
            raise ValueError('embeddings should be path to FastText file or '
                             'gensim FastTextKeyedVectors object or None'
                             f'got {type(embeddings)} instead')
        self.embeddings = embeddings

        if isinstance(embeddings, str):
            print('Loading embeddings from file')
            self.embeddings = load_facebook_vectors(embeddings)

        self.max_text_len = max_text_len
        self.unk_vec = np.random.rand(self.embeddings.vector_size)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        _text_len = min(self.max_text_len, len(item.text))
        _text_tensor = torch.zeros([_text_len, self.embeddings.vector_size])

        # TODO: tokenize after noising
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


class KeyedVectorsCSVDataset(torch.utils.data.Dataset):
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
            self.embeddings = load_facebook_vectors(embeddings)
        elif isinstance(embeddings, (FastTextKeyedVectors, KeyedVectors)):
            self.embeddings = embeddings
        elif embeddings is None:
            self.embeddings = None
            assert elmo
        else:
            raise ValueError('embeddings should be path to FastText file or '
                             'gensim FastTextKeyedVectors object or None'
                             f'got {type(embeddings)} instead')

        self.alphabet = alphabet or cfg.alphabet
        self.elmo = elmo
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
        if not self.elmo: text = self._preprocess(text)  # noqa E701
        return text, label

    def _tokenize(self, text):
        return word_tokenize(text)

    def _preprocess(self, text):
        _text_len = min(self.max_text_len, len(text))
        _text_tensor = torch.zeros([_text_len, self.embeddings.vector_size],
                                   dtype=torch.float32)

        for i, token in enumerate(text):
            if i >= self.max_text_len: break  # noqa: E701

            # KeyedVectors object does not have .get() method
            if token in self.embeddings:
                token_vec = self.embeddings[token]
            else:
                token_vec = self.unk_vec

            token_tensor = torch.FloatTensor(token_vec)
            _text_tensor[i, :] = token_tensor

        return _text_tensor

    def _noise_generator(self, string):
        return noise_generator(string, self.noise_level, self.alphabet)


class ALaCarteCSVDataset(KeyedVectorsCSVDataset):
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
                 induce_vectors=False,
                 window_half_size=10,
                 induction_iterations=1,
                 induction_matrix=None):
        """
        Induce à la carte embeddings for unk words
        if there are multiple unk words in a text
        [he, was, so, unk, that, he, unk, unk]
        multiple induction iterations might help (or might not)

        :param induction_matrix: np.array with shape (embedding_dim, embedding_dim)
            or 'identity' for just averaging the vectors
        """

        super().__init__(
            filepath, text_field, label_field, embeddings, max_text_len, alphabet, elmo=False
        )
        assert induce_vectors == bool(induction_matrix), 'induce_vectors and induction_matrix should both be specified'
        if isinstance(induction_matrix, str) and induction_matrix == 'identity':
            induction_matrix = np.identity(self.embeddings.vector_size)
        if induction_matrix is not None:
            assert induction_matrix.shape[0] == induction_matrix.shape[1] == self.embeddings.vector_size

        self.induce_vectors = induce_vectors
        self.induction_matrix = induction_matrix

        self.window_half_size = window_half_size
        self.induction_iterations = induction_iterations
        self.sort_key = None

    def _preprocess(self, text):
        """
        :param text: tokenized text, list(str)
        """
        _text_len = min(self.max_text_len, len(text))
        _text_tensor = np.zeros([_text_len, self.embeddings.vector_size], dtype=np.float32)
        self.sort_key = _text_len

        unk_indices = []

        for i, token in enumerate(text):
            if i >= self.max_text_len: break  # noqa: E701

            if token in self.embeddings.vocab:  # do not use fastText OOV
                token_vec = self.embeddings[token]
                _text_tensor[i, :] = token_vec
            else:
                # mark unk
                unk_indices.append(i)
                # token_vec = self.unk_vec
                # _text_tensor[i, :] = token_vec

        if self.induce_vectors:
            _text_tensor = self._induce_vectors(_text_tensor, unk_indices, _text_len)

        _text_tensor = torch.tensor(_text_tensor)

        return _text_tensor

    def _induce_vectors(self, _text_tensor, unk_indices, _text_len):
        # import pdb; pdb.set_trace()
        for _ in range(self.induction_iterations):
            for i in unk_indices:
                left_word = max(0, i - self.window_half_size)
                right_word = min(_text_len, i + self.window_half_size + 1)
                context_indices = [v for v in list(range(left_word, right_word)) if v not in unk_indices]
                context_vector = self.unk_vec
                if len(context_indices):
                    context_vectors = _text_tensor[context_indices, :]
                    context_vector = np.mean(context_vectors, 0)
                induced_embedding = self.induction_matrix @ context_vector
                _text_tensor[i, :] = induced_embedding
        return _text_tensor
