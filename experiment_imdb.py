"""
Main experiment script
"""
import argparse
from time import time

import pandas as pd

import torch
import torchtext

from gensim.models import FastText

import cfg
from train import train
from text_classification.datautils import CharIMDB, FastTextIMDB, HierarchicalIMDB
from text_classification.layers import CharCNN, RNNBinaryClassifier, YoonKimModel, AttentionedYoonKimModel
from text_classification import trainutils
from text_classification.logger import logger


MAXLEN = 512
# MODEL_TYPE = 'FastText'

parser = argparse.ArgumentParser()
parser.add_argument('--model-name')


def experiment(model_class, train_data, test_data, **model_params):
    train_dataloader, val_dataloader, test_dataloader = \
        trainutils.get_dataloaders(train_data, test_data, batch_size=cfg.train.batch_size,
                                   valid_size=cfg.train.val_size)

    all_results = []
    for i, noise_level in enumerate(cfg.experiment.noise_levels):
        logger.info('Training model for noise level {:.3f} ({}/{})'
                    .format(noise_level, i, len(cfg.experiment.noise_levels)))

        model = model_class(**model_params)

        # TODO: make this device-agnostic
        with torch.cuda.device(0):
            trained_model, results = train(model,
                                           train_dataloader,
                                           val_dataloader,
                                           test_dataloader,
                                           test_dataloader,
                                           noise_level,
                                           comment='test',
                                           save_model_path='models',
                                           save_results_path='results/CharCNN_IMDB.csv')
            all_results.append(results)

    pd.DataFrame(all_results).to_csv('results/CharCNN_IMDB.csv')
    logger.info('Total execution time: {:.1f}min'.format((time() - start_time) / 60.))


if __name__ == '__main__':
    start_time = time()
    logger.info('The script is started')

    args = parser.parse_args()

    if cfg.train.evals_per_noise_level == 1:
        logger.warning('Only one eval for noise level on test!')

    if not cfg.cuda:
        logger.warning('Not using CUDA!')

    # Chose data format
    if args.model_name == 'CharCNN':
        text_field = torchtext.data.Field(
            lower=True, include_lengths=False, tensor_type=torch.FloatTensor, batch_first=True,
            tokenize=lambda x: x, use_vocab=False, sequential=False
        )
        label_field = torchtext.data.Field(sequential=False, use_vocab=False)

    else:
        text_field = torchtext.data.Field(
            lower=True, include_lengths=False, tensor_type=torch.FloatTensor, batch_first=True,
            tokenize='spacy', use_vocab=False
        )
        label_field = torchtext.data.Field(sequential=False, use_vocab=False)

    # Chose model
    logger.info('Creating datasets and preprocessing raw texts...')
    if args.model_name == 'CharCNN':
        CharIMDB.maxlen = MAXLEN
        train_data, test_data = CharIMDB.splits(text_field, label_field)

        model_class = CharCNN
        model_params = {'n_filters': 256, 'cnn_kernel_size': 15, 'maxlen': MAXLEN, 'alphabet_len': len(cfg.alphabet)}

    elif args.model_name == 'FastText':
        logger.info('Loading embeddings...')
        embeddings = FastText.load_fasttext_format(cfg.data.fasttext_path)
        train_data, test_data = FastTextIMDB.splits(text_field, label_field, embeddings=embeddings)

        model_class = RNNBinaryClassifier
        model_params = {'input_dim': embeddings.vector_size, 'hidden_dim': 256}

    elif args.model_name == 'YoonKim':
        train_data, test_data = HierarchicalIMDB.splits(text_field, label_field)

        model_class = YoonKimModel
        model_params = {'n_filters': 256, 'cnn_kernel_size': 5, 'hidden_dim_out': 128}

    elif args.model_name == 'AttentionedYoonKim':
        train_data, test_data = HierarchicalIMDB.splits(text_field, label_field)

        model_class = AttentionedYoonKimModel
        model_params = {'n_filters': 256, 'cnn_kernel_size': 5, 'hidden_dim_out': 128, 'heads': 1}

    else:
        raise ValueError('Wrong MODEL_TYPE')

    logger.info('Starting the experiment')
    experiment(model_class, train_data, test_data, **model_params)
