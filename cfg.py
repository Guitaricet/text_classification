import numpy as np
import torch


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

alphabet = ['<PAD>', '<UNK>'] + [s for s in """ abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'’/\|_@#$%ˆ&*‘+-=<>()[]{}"""]
russian_chars = [s for s in 'абвгдеёжзийклмнопрстуфхцчщъыьэюя']

max_word_len = 8
max_text_len = 256

pin_memory = False

elmo = False


class data:
    fasttext_path = '/data/embeddings/wiki.simple.bin'
    # word2vec_path =
    elmo_options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"  # noqa
    elmo_weights_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"  # noqa


class train:
    batch_size = 32
    val_size = 0.15
    val_every = 1000
    log_every = 10
    num_workers = 6
    evals_per_noise_level = 1
    lr = 1e-3
    epochs = 20


class experiment:
    noise_levels = list(reversed(np.concatenate([np.arange(0, 0.05, 0.01), np.arange(0.05, 0.2, 0.025), [0.9]])))
    n_trains = 10
