"""
Main experiment script for .csv datasets
Currently mokoron sentiment analysis dataset and airline tweets.
"""
import os
import argparse
from time import time, sleep

import pandas as pd

from torch.utils.data import DataLoader

from gensim.models import FastText

import cfg
from train import train, evaluate_on_noise
from text_classification import trainutils
from text_classification.logger import logger
from text_classification.layers import CharCNN, RNNBinaryClassifier, YoonKimModel, AttentionedYoonKimModel
from text_classification.datautils import CharMokoron, FastTextMokoron, HierarchicalMokoron


MAXLEN = 170  # for CharCNN
cfg.max_text_len = 128

parser = argparse.ArgumentParser()
parser.add_argument('--model-name')
parser.add_argument('--dataset-name', choices=['mokoron', 'airline-tweets'])
parser.add_argument('--comment', default='')
parser.add_argument('--datapath', default='data/mokoron')


def experiment(model_class, train_data, val_data, test_data, test_original_data,
               save_results_path, comment, lr, epochs, **model_params):
    train_dataloader, val_dataloader, test_dataloader = \
        trainutils.get_dataloaders(train_data, test_data, validset=val_data, batch_size=cfg.train.batch_size)
    test_original_dataloader = DataLoader(test_original_data,
                                          batch_size=cfg.train.batch_size,
                                          num_workers=cfg.train.num_workers,
                                          pin_memory=cfg.pin_memory)

    noise_levels = cfg.experiment.noise_levels
    all_results = pd.DataFrame()

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
                              save_model_path='models')

        logger.info('Calculating test metrics... Absolute time T={:.2f}min'.format((time() - start_time) / 60.))
        sleep(2)  # workaround from ConnectionResetError
        # https://stackoverflow.com/questions/47762973/python-pytorch-multiprocessing-throwing-errors-connection-reset-by-peer-and-f
        model.eval()
        train_metrics = trainutils.get_metrics(trained_model, train_dataloader, frac=0.1)
        results_dicts_noised = evaluate_on_noise(trained_model, test_dataloader, noise_levels, cfg.train.evals_per_noise_level)
        results_dicts_original = evaluate_on_noise(trained_model, test_original_dataloader, [0], 1)

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
    start_time = time()
    logger.info('The script is started')

    args = parser.parse_args()

    if cfg.train.evals_per_noise_level == 1:
        logger.warning('Only one eval for noise level on test!')

    if not cfg.cuda:
        logger.warning('Not using CUDA!')

    basepath = args.datapath.strip('/') + '/'
    if args.dataset_name == 'mokoron':
        text_filed = 'text_spellchecked'
        text_original_field = 'text_original'
        label_field = 'sentiment'

        alphabet = cfg.alphabet + cfg.russian_chars
        alphabet = [c for c in alphabet if c not in ('(', ')')]
    elif args.dataset_name == 'airline-tweets':
        raise NotImplementedError
    else:
        raise ValueError('Incorrect dataset name')

    # Chose data
    logger.info('Creating datasets and preprocessing raw texts...')

    if args.model_name == 'CharCNN':
        CharMokoron.maxlen = MAXLEN
        CharMokoron.alphabet = alphabet
        train_data = CharMokoron(basepath + 'train.csv', text_filed, label_field)
        valid_data = CharMokoron(basepath + 'validation.csv', text_filed, label_field)
        test_data = CharMokoron(basepath + 'test.csv', text_filed, label_field)

        test_original_data = CharMokoron(basepath + 'test.csv', text_original_field, label_field)

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
        embeddings = FastText.load_fasttext_format(cfg.data.fasttext_path)
        train_data = FastTextMokoron(basepath + 'train.csv', text_filed, label_field, embeddings)
        valid_data = FastTextMokoron(basepath + 'validation.csv', text_filed, label_field, embeddings)
        test_data = FastTextMokoron(basepath + 'test.csv', text_filed, label_field, embeddings)

        test_original_data = FastTextMokoron(basepath + 'test.csv', text_original_field, label_field, embeddings)

        model_class = RNNBinaryClassifier
        model_params = {'input_dim': embeddings.vector_size, 'hidden_dim': 256, 'dropout': 0.5}
        lr = 0.0006
        epochs = 20

    elif args.model_name == 'YoonKim':
        HierarchicalMokoron.alphabet = alphabet
        train_data = HierarchicalMokoron(basepath + 'train.csv', text_filed, label_field)
        valid_data = HierarchicalMokoron(basepath + 'validation.csv', text_filed, label_field)
        test_data = HierarchicalMokoron(basepath + 'test.csv', text_filed, label_field)

        test_original_data = HierarchicalMokoron(basepath + 'test.csv', text_original_field, label_field)

        model_class = YoonKimModel
        model_params = {'n_filters': 32,
                        'cnn_kernel_size': 5,
                        'hidden_dim_out': 64,
                        'embedding_dim': 90,
                        'dropout': 0.5}
        lr = 1e-3
        epochs = 20

    elif args.model_name == 'AttentionedYoonKim':
        HierarchicalMokoron.alphabet = alphabet
        train_data = HierarchicalMokoron(basepath + 'train.csv', text_filed, label_field)
        valid_data = HierarchicalMokoron(basepath + 'validation.csv', text_filed, label_field)
        test_data = HierarchicalMokoron(basepath + 'test.csv', text_filed, label_field)

        test_original_data = HierarchicalMokoron(basepath + 'test.csv', text_original_field, label_field)

        model_class = AttentionedYoonKimModel
        model_params = {'n_filters': 128,
                        'cnn_kernel_size': 5,
                        'hidden_dim_out': 128,
                        'embedding_dim': 74,
                        'heads': 1}
        lr = 1e-3
        epochs = 20

    else:
        raise ValueError('Wrong model name')

    save_results_path = 'results/%s_%s.csv' % (args.model_name, args.dataset_name)
    if os.path.exists(save_results_path):
        if input('File at path %s already exists, delete it? (y/n)').lower() != 'y':
            logger.warning('Cancelling execution due to existing output file')
            exit(1)

    logger.info('Starting the experiment')
    experiment(model_class,
               train_data,
               valid_data,
               test_data,
               test_original_data,
               save_results_path=save_results_path,
               comment=args.comment,
               lr=lr,
               epochs=epochs,
               **model_params)
    logger.info('Total execution time: {:.1f}min'.format((time() - start_time) / 60.))
