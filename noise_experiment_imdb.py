"""
Main experiment script for IMDB dataset
"""
import os
import argparse
from time import time, sleep

import pandas as pd

import torch
import torchtext

from gensim.models import FastText

import cfg
from train import train, evaluate_on_noise
from text_classification import trainutils
from text_classification.logger import logger
from text_classification.modules import CharCNN, RNNClassifier, YoonKimModel
from text_classification.datautils import CharIMDB, FastTextIMDB, HierarchicalIMDB


parser = argparse.ArgumentParser()
parser.add_argument('--model-name')
parser.add_argument('--comment', default='')


MAX_TEXT_LEN = 256


def experiment(model_class, train_data, test_data,
               save_results_path, comment, lr, epochs, **model_params):
    train_dataloader, val_dataloader, test_dataloader = \
        trainutils.get_dataloaders(train_data, test_data, batch_size=cfg.train.batch_size,
                                   valid_size=cfg.train.val_size)

    noise_levels = cfg.experiment.noise_levels
    all_results = pd.DataFrame()

    for _ in range(cfg.experiment.n_trains):
        for i, noise_level in enumerate(noise_levels):
            logger.info('Training model for noise level {:.3f} ({}/{})'
                        .format(noise_level, i, len(noise_levels)))

            model = model_class(**model_params)

            trained_model = train(model,
                                  train_dataloader,
                                  val_dataloader,
                                  noise_level,
                                  lr=lr,
                                  epochs=epochs,
                                  comment=comment,
                                  log_every=cfg.train.log_every,
                                  save_model_path='models')

            logger.info('Calculating test metrics... Absolute time T={:.2f}min'.format((time() - start_time) / 60.))
            sleep(2)  # workaround for ConnectionResetError
            # https://stackoverflow.com/questions/47762973/python-pytorch-multiprocessing-throwing-errors-connection-reset-by-peer-and-f
            model.eval()
            train_metrics = trainutils.get_metrics(trained_model, train_dataloader, frac=0.1)
            results_dicts_noised = evaluate_on_noise(trained_model, test_dataloader, noise_levels, cfg.train.evals_per_noise_level)
            results_dicts_original = evaluate_on_noise(trained_model, test_dataloader, [0], 1)

            results_df_noised = pd.DataFrame(results_dicts_noised)
            results_df_original = pd.DataFrame(results_dicts_original)
            results_df_original['noise_level_test'] = -1
            results_df = pd.concat([results_df_noised, results_df_original], sort=False)

            results_df['model_type'] = trained_model.name
            results_df['noise_level_train'] = noise_level
            results_df['acc_train'] = train_metrics['accuracy']
            results_df['f1_train'] = train_metrics['f1']
            all_results = pd.concat([all_results, results_df], sort=False)
            logger.info('Saving the results')
            all_results.to_csv(save_results_path)

    all_results.to_csv(save_results_path)


if __name__ == '__main__':
    """
    IMDB
    """
    MAXLEN = 512  # for CharCNN

    args = parser.parse_args()

    save_results_path = 'results/%s_IMDB.csv' % args.model_name
    if os.path.exists(save_results_path):
        if input('File at path %s already exists, delete it? (y/n)' % save_results_path).lower() != 'y':
            logger.warning('Cancelling execution due to existing output file')
            exit(1)

    start_time = time()
    logger.info('The script is started')

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
        model_params = {'n_filters': 128,
                        'cnn_kernel_size': 5,
                        'dropout': 0.5,
                        'maxlen': MAXLEN,
                        'alphabet_len': len(cfg.alphabet)}
        lr = 1e-3
        epochs = 30

    elif args.model_name == 'FastText':
        logger.info('Loading embeddings...')
        logger.info('maxlen: %s' % cfg.max_text_len)
        embeddings = FastText.load_fasttext_format(cfg.data.fasttext_path)
        train_data, test_data = FastTextIMDB.splits(text_field, label_field, embeddings=embeddings)

        model_class = RNNClassifier
        model_params = {'input_dim': embeddings.vector_size, 'hidden_dim': 256, 'dropout': 0.2}
        lr = 0.0006
        epochs = 20

    elif args.model_name == 'YoonKim':
        train_data, test_data = HierarchicalIMDB.splits(text_field, label_field)

        model_class = YoonKimModel
        model_params = {'n_filters': 32,
                        'cnn_kernel_size': 5,
                        'hidden_dim_out': 64,
                        'embedding_dim': 90,
                        'dropout': 0.5}
        lr = 1e-3
        epochs = 20

    else:
        raise ValueError('Wrong model name')

    logger.info('Starting the experiment')
    experiment(model_class,
               train_data,
               test_data,
               save_results_path=save_results_path,
               comment=args.comment,
               lr=lr,
               epochs=epochs,
               **model_params)
    logger.info('Total execution time: {:.1f}min'.format((time() - start_time) / 60.))
