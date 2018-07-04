import numpy as np
import torch


cuda = torch.cuda.is_available()

# TODO: move this to class data
alphabet = ['<UNK>', '\n'] + [s for s in """ abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'’’/\|_@#$%ˆ&* ̃‘+-=<>()[]{}"""]
max_word_len = 8
max_text_len = 256


class data:
    fasttext_path = '~/Downloads/wiki.simple.bin'
    # word2vec_path =


class train:
    batch_size = 32
    val_size = 0.15
    num_workers = 4
    evals_per_noise_level = 1


class experiment:
    noise_levels = np.concatenate([np.arange(0.05, 0.2, 0.01), np.arange(0, 0.05, 0.005)])
