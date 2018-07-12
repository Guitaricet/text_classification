"""
For RoVe we make a lot of files for each noise level
"""
import re
import os
from glob import iglob as glob
from random import random, choice

import numpy as np
import pandas as pd

from tqdm import tqdm


ALPHABET = [s for s in """ abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'’’/\|_@#$%ˆ&* ̃‘+-=<>()[]{}"""]
RUSSIAN_CHARS = [s for s in 'абвгдеёжзийклмнопрстуфхцчщъыьэюя']
NOISE_LEVELS = np.concatenate([np.arange(0.05, 0.2, 0.01), np.arange(0, 0.05, 0.005)])


def noise_generator(string, noise_level, alphabet):
    noised = ""
    for c in string:
        if random() > noise_level:
            noised += c
        if random() < noise_level:
            noised += choice(alphabet)
    return noised


def noise_csv_dataset(basepath, lang):
    alphabet = ALPHABET
    if lang == 'ru':
        alphabet += RUSSIAN_CHARS

    for ds_type in ['train', 'validation', 'test']:
        data = pd.read_csv(os.path.join(basepath, '%s.csv' % ds_type))
        texts = data['text_original']
        print('Saving original texts')
        savepath = os.path.join(basepath, 'rove/original/%s.txt' % ds_type)
        os.makedirs(os.path.dirname(savepath), exist_ok=True)

        with open(savepath, 'w') as f:
            for text in texts:
                f.write(re.sub('\n', ' ', text) + '\n')

        print('Saving noised texts')
        texts = data['text_spellchecked']
        for noise_level in tqdm(NOISE_LEVELS):
            savepath = os.path.join(basepath, 'rove/noised_{:.3f}:/{}.txt'.format(noise_level, ds_type))
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            with open(savepath, 'w') as f:
                for text in texts:
                    noised_text = noise_generator(re.sub('\n', ' ', text), noise_level, alphabet)
                    f.write(noised_text + '\n')


def noise_imdb():
    ...
    # for fpath in glob('../.data/')


if __name__ == '__main__':
    # noise_csv_dataset('../data/mokoron', lang='ru')
    # noise_csv_dataset('../data/airline_tweets', lang='en')
    noise_imdb()
