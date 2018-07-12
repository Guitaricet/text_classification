import numpy as np
import torch


cuda = torch.cuda.is_available()

# TODO: add russian chars
# TODO: move this to class data
alphabet = ['<UNK>', '\n'] + [s for s in """ abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'’’/\|_@#$%ˆ&* ̃‘+-=<>()[]{}"""]
russian_chars = [s for s in 'абвгдеёжзийклмнопрстуфхцчщъыьэюя']

max_word_len = 8
max_text_len = 256

pin_memory = True

class data:
    fasttext_path = '/data/embeddings/wiki.simple.bin'
    # word2vec_path =


class train:
    batch_size = 32
    val_size = 0.15
    num_workers = 4
    evals_per_noise_level = 10
    lr = 1e-3
    epochs = 20


class experiment:
    noise_levels = np.concatenate([np.arange(0, 0.05, 0.01), np.arange(0.05, 0.2, 0.025)])
