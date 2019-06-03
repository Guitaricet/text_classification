import re

import numpy as np
import pandas as pd

import torch

from nltk.tokenize import word_tokenize
from gensim.models.keyedvectors import FastTextKeyedVectors, KeyedVectors
from gensim.models.fasttext import load_facebook_vectors


import cfg
from text_classification.utils import noise_generator


class AbstractNoisedDataset(torch.utils.data.Dataset):

    def set_noise_level(self, noise_level, force_renoise=False):
        if noise_level != self.noise_level and not force_renoise:
            self._noise_level = noise_level
            self._data = self._preprocess_df(self.data)

    def _preprocess_text(self, text):
        text = text.lower()
        # for mokoron dataset we should remove smiles
        if ')' not in self.alphabet:
            text = re.sub('[(,)]', '', text)
        if self.noise_level > 0:
            text = self._noise_generator(text)
        text = self._tokenize(text)
        return text

    def _preprocess_df(self, df):
        df[self.label_field] = df[self.label_field].apply(lambda x: self.label2int[x])
        df[self.text_field] = df[self.text_field].apply(self._preprocess_text)
        return df

    def __getitem__(self, idx):
        # processing from char representation is required for text noising
        line = self.data.iloc[idx]
        text = line[self.text_field]
        label = line[self.label_field]

        text = self._numericalize(text)
        return text, label

    def _noise_generator(self, string):
        return noise_generator(string, self.noise_level, self.alphabet)

    def _tokenize(self, text):
        return word_tokenize(text)


class HierarchicalCSVDataset(AbstractNoisedDataset):
    """
    Dataset class for hierarchical (chars -> words -> text) networks which reads data from .csv

    Mokoron, because it was firstly used for Mokoron twitter sentiment dataset
    Zero vector used for padding
    """

    def __init__(self,
                 filepath,
                 text_field,
                 label_field,
                 max_word_len=cfg.max_word_len,
                 max_text_len=cfg.max_text_len,
                 noise_level=0,
                 alphabet=None):

        self._noise_level = noise_level
        self.data = pd.read_csv(filepath)
        self.alphabet = alphabet or cfg.alphabet
        self.text_field = text_field
        self.label_field = label_field
        self.max_word_len = max_word_len
        self.max_text_len = max_text_len
        self.char2int = {s: i for i, s in enumerate(self.alphabet)}
        self.unk_index = self.char2int['<UNK>']
        self.pad_index = self.char2int['<PAD>']
        self.label2int = {l: i for i, l in enumerate(sorted(self.data[self.label_field].unique()))}
        self.data = self._preprocess_df(self.data)

    @property
    def noise_level(self):
        return self._noise_level

    def __len__(self):
        return len(self.data)

    def _numericalize(self, text):
        _text_len = min(self.max_text_len, len(text))
        _text_tensor = torch.zeros([_text_len, self.max_word_len], dtype=torch.long)

        for i, token in enumerate(text):
            if i >= self.max_text_len: break  # noqa: E701
            char_ids = [self.char2int.get(c, self.unk_index) for c in token]
            _text_tensor[i, :len(char_ids)] = char_ids

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


class KeyedVectorsCSVDataset(AbstractNoisedDataset):
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
                 noise_level=0,
                 elmo=False):
        assert not elmo, 'ELMo support is deprecated'
        if isinstance(embeddings, str):
            self.embeddings = load_facebook_vectors(embeddings)
        elif isinstance(embeddings, (FastTextKeyedVectors, KeyedVectors)):
            self.embeddings = embeddings
        else:
            raise ValueError('embeddings should be path to FastText file or '
                             'gensim FastTextKeyedVectors object'
                             f'got {type(embeddings)} instead')

        self._noise_level = noise_level
        self.alphabet = alphabet or cfg.alphabet
        self.text_field = text_field
        self.label_field = label_field
        self.data = pd.read_csv(filepath)
        self.max_text_len = max_text_len
        if self.embeddings is not None:
            self.unk_vec = np.random.rand(self.embeddings.vector_size)
        self.label2int = {l: i for i, l in enumerate(sorted(self.data[self.label_field].unique()))}
        self._data = self._preprocess_df(self.data)

    @property
    def noise_level(self):
        return self._noise_level

    def __len__(self):
        return len(self._data)

    def _numericalize(self, text):
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
                 noise_level=0,
                 induce_vectors=False,
                 window_half_size=2,
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
            filepath, text_field, label_field, embeddings, max_text_len, alphabet, noise_level
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

    def _numericalize(self, text):
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
