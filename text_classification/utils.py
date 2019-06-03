from functools import partialmethod
from random import random, choice
import torch

import cfg


def swap(c, i, j):
    c = list(c)
    c[i], c[j] = c[j], c[i]
    return ''.join(c)


def noise_generator(text, noise_level, alphabet):
    noised = ""
    for c in text:
        if random() > noise_level:
            noised += c
        if random() < noise_level:
            noised += choice(alphabet)
        if random() < noise_level and len(noised) > 1:
            noised = swap(noised, -1, -2)

    return noised


def get_dataset_params(dataset_name, train_on_original_data):
    alphabet = cfg.alphabet
    text_field = 'text_spellchecked'

    if dataset_name == 'mokoron':
        text_field_original = 'text_original'
        label_field = 'sentiment'

        alphabet = cfg.alphabet + cfg.russian_chars
        alphabet = [c for c in alphabet if c not in ('(', ')')]

        n_classes = 2

    elif dataset_name == 'airline-tweets':
        text_field_original = 'text_original'
        label_field = 'airline_sentiment'
        n_classes = 3

    elif dataset_name == 'airline-tweets-binary':
        text_field_original = 'text_original'
        label_field = 'airline_sentiment'
        n_classes = 2

    elif dataset_name == 'rusentiment':
        text_field_original = 'text'
        label_field = 'label'
        n_classes = 5

    elif dataset_name == 'sentirueval':
        text_field_original = 'text'
        label_field = 'label'
        n_classes = 4

    else:
        raise ValueError('Incorrect dataset name')

    if train_on_original_data:
        text_field = text_field_original

    return text_field, text_field_original, label_field, n_classes, alphabet


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size, dtype=vec.dtype)], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len = max([x.shape[self.dim] for x, _ in batch])
        # pad according to max_len
        batch = [(pad_tensor(x, pad=max_len, dim=self.dim), y) for x, y in batch]
        # stack all
        xs = torch.stack([x[0] for x in batch], dim=0)
        ys = torch.LongTensor([x[1] for x in batch])
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)

def partialclass(cls, *args, **kwds):

    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwds)

    return NewCls
