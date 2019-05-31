from random import random, choice


def noise_generator(text, noise_level, alphabet):
    noised = ""
    for c in text:
        if random() > noise_level:
            noised += c
        if random() < noise_level:
            noised += choice(alphabet)
    return noised
