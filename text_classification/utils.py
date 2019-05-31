from random import random, choice

import cfg


def noise_generator(text, noise_level, alphabet):
    noised = ""
    for c in text:
        if random() > noise_level:
            noised += c
        if random() < noise_level:
            noised += choice(alphabet)
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
