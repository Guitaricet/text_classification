"""
Main experiment script for .csv datasets
Currently mokoron sentiment analysis dataset and airline tweets.
"""
import os
import argparse
from time import time, sleep

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors

import cfg
from train import train, evaluate_on_noise
from text_classification import utils, trainutils
from text_classification.utils import PadCollate, partialclass
from text_classification.logger import logger
from text_classification.modules import RNNClassifier, YoonKimModel
from text_classification.datautils import KeyedVectorsCSVDataset, HierarchicalCSVDataset, ALaCarteCSVDataset

from allennlp.modules.elmo import Elmo
# TODO: add run name to results csv


parser = argparse.ArgumentParser()
parser.add_argument('--model-name')
parser.add_argument('--dataset-name')
parser.add_argument('--comment', default='')
parser.add_argument('--datapath', default='data/mokoron')
parser.add_argument('--noise-level', type=float, default=None)
parser.add_argument('--embeddings-path', default=None)
parser.add_argument('-y', default=False, action='store_true', help='yes to all')
parser.add_argument('--original-train', default=False, action='store_true', help='train_on_original_dataset')  # noqa E501
parser.add_argument('--sample-data', type=float, default=1.0)
parser.add_argument('--induction-matrix', type=str, help='path to a la carte tranform_matrix.bin or string "identity"')


def experiment(model_class, train_data, val_data, test_data, test_original_data,
               save_results_path, comment, lr, epochs, noise_level=None, **model_params):
    """
    Results columns:
    ,acc_test,f1_test,noise_level_test,model_type,noise_level_train,acc_train,f1_train
    """
    train_dataloader, val_dataloader, test_dataloader = \
        trainutils.get_dataloaders(train_data, test_data, validset=val_data, batch_size=cfg.train.batch_size)  # noqa E501
    test_original_dataloader = DataLoader(test_original_data,
                                          batch_size=cfg.train.batch_size,
                                          num_workers=cfg.train.num_workers,
                                          pin_memory=cfg.pin_memory,
                                          collate_fn=PadCollate(0))
    if noise_level is not None:
        noise_levels = [noise_level]
    else:
        noise_levels = cfg.experiment.noise_levels
    all_results = pd.DataFrame()

    for _ in range(cfg.experiment.n_trains):
        for i, noise_level in enumerate(noise_levels):
            logger.info('Training model for noise level {:.3f} ({}/{})'
                        .format(noise_level, i+1, len(noise_levels)))

            model = model_class(**model_params)

            model = train(model,
                          train_dataloader,
                          val_dataloader,
                          noise_level,
                          lr=lr,
                          epochs=epochs,
                          comment=comment)

            logger.info('Calculating test metrics... Absolute time T={:.2f}min'.format((time() - start_time) / 60.))  # noqa E501
            sleep(2)  # workaround for ConnectionResetError
            # https://stackoverflow.com/questions/47762973/python-pytorch-multiprocessing-throwing-errors-connection-reset-by-peer-and-f
            model.eval()
            train_metrics = trainutils.get_metrics(model, train_dataloader, frac=0.1)
            results_dicts_noised = evaluate_on_noise(model, test_dataloader, [noise_level], cfg.train.evals_per_noise_level)  # noqa E501
            results_dicts_original = evaluate_on_noise(model, test_original_dataloader, [0], 1)  # noqa E501
            # for testing
            evaluate_on_noise(model, val_dataloader, [noise_level], cfg.train.evals_per_noise_level)  # noqa E501

            results_df_noised = pd.DataFrame(results_dicts_noised)
            results_df_original = pd.DataFrame(results_dicts_original)
            results_df_original['noise_level_test'] = -1
            results_df = pd.concat([results_df_noised, results_df_original], sort=False)

            results_df['model_type'] = model.name
            results_df['noise_level_train'] = noise_level
            results_df['acc_train'] = train_metrics['accuracy']
            results_df['f1_train'] = train_metrics['f1']
            all_results = pd.concat([all_results, results_df], sort=False)
            logger.info('Saving the results')
            all_results.to_csv(save_results_path)

    all_results.to_csv(save_results_path)


if __name__ == '__main__':
    """
    Tweets
    """
    max_text_len = 64

    args = parser.parse_args()

    save_results_path = f'results/{args.model_name}_{args.dataset_name}{args.comment}.csv'
    if args.original_train:
        save_results_path += '_orig'
    if os.path.exists(save_results_path) and not args.y:
        if input('File at path %s already exists, delete it? (y/n) ' % save_results_path).lower() != 'y':  # noqa E501
            logger.warning('Cancelling execution due to existing output file')
            exit(1)

    start_time = time()
    logger.info('The script is started')

    if cfg.train.evals_per_noise_level == 1:
        logger.warning('Only one eval for noise level on test!')

    if not cfg.device == 'cuda':
        logger.warning('Not using CUDA!')

    basepath = args.datapath.rstrip('/') + '/'
    dataset_name = args.dataset_name.lower()

    text_field, text_field_original, label_field, n_classes, alphabet =\
        utils.get_dataset_params(dataset_name, args.original_train)

    # Chose data
    logger.info('Creating datasets and preprocessing raw texts...')

    if args.model_name == 'FastText':
        logger.info('Loading embeddings...')
        embeddings = load_facebook_vectors(args.embeddings_path or cfg.data.fasttext_path)
        get_dataset = partialclass(KeyedVectorsCSVDataset,
                                   label_field=label_field,
                                   embeddings=embeddings,
                                   alphabet=alphabet,
                                   max_text_len=max_text_len)

        train_data = get_dataset(basepath + 'train.csv', text_field)
        valid_data = get_dataset(basepath + 'validation.csv', text_field)
        test_data = get_dataset(basepath + 'test.csv', text_field)

        test_original_data = get_dataset(basepath + 'test.csv', text_field_original)

        model_class = RNNClassifier
        model_params = {'input_dim': embeddings.vector_size, 'hidden_dim': 256, 'dropout': 0.5,
                        'num_classes': n_classes}
        lr = 0.0006
        epochs = 20

    elif args.model_name.lower() == 'alacarte':
        logger.info('Loading embeddings...')
        embeddings = KeyedVectors.load_word2vec_format(args.embeddings_path)
        induction_matrix = args.induction_matrix.lower()
        if induction_matrix != 'identity' and induction_matrix is not None:
            induction_matrix = np.fromfile(induction_matrix, dtype=np.float32)
            if len(induction_matrix) != 2:
                d = int(np.sqrt(induction_matrix.shape[0]))
                induction_matrix = induction_matrix.reshape(d, d)
        get_dataset = partialclass(ALaCarteCSVDataset,
                                   label_field=label_field,
                                   embeddings=embeddings,
                                   alphabet=alphabet,
                                   max_text_len=max_text_len,
                                   induce_vectors=induction_matrix is not None,
                                   induction_matrix=induction_matrix)

        train_data = get_dataset(basepath + 'train.csv', text_field)
        valid_data = get_dataset(basepath + 'validation.csv', text_field)
        test_data = get_dataset(basepath + 'test.csv', text_field)

        test_original_data = get_dataset(basepath + 'test.csv', text_field_original)

        model_class = RNNClassifier
        model_params = {'input_dim': embeddings.vector_size, 'hidden_dim': 256, 'dropout': 0.5,
                        'num_classes': n_classes}
        lr = 0.0006
        epochs = 20

    elif args.model_name == 'ELMo':
        raise RuntimeError('ELMo is broken')
        logger.info('Loading embeddings...')

        elmo = Elmo(cfg.data.elmo_options_file, cfg.data.elmo_weights_file, 1, dropout=0)
        get_dataset = partialclass(KeyedVectorsCSVDataset,
                                   label_field=label_field,
                                   embeddings=embeddings,
                                   alphabet=alphabet,
                                   max_text_len=max_text_len,
                                   elmo=True)

        train_data = get_dataset(basepath + 'train.csv', text_field)
        valid_data = get_dataset(basepath + 'validation.csv', text_field)
        test_data = get_dataset(basepath + 'test.csv', text_field)

        test_original_data = get_dataset(basepath + 'test.csv', text_field_original)

        model_class = RNNClassifier
        model_params = {'input_dim': 1024, 'hidden_dim': 256, 'dropout': 0.5,
                        'num_classes': n_classes, 'elmo': elmo}
        lr = 0.0006
        epochs = 10

    elif args.model_name == 'YoonKim':
        get_dataset = partialclass(HierarchicalCSVDataset,
                                   label_field=label_field,
                                   alphabet=alphabet,
                                   max_text_len=max_text_len)
        train_data = get_dataset(basepath + 'train.csv', text_field)
        valid_data = get_dataset(basepath + 'validation.csv', text_field)
        test_data = get_dataset(basepath + 'test.csv', text_field)

        test_original_data = get_dataset(basepath + 'test.csv', text_field_original)

        model_class = YoonKimModel
        model_params = {'n_filters': 32,
                        'cnn_kernel_size': 5,
                        'hidden_dim_out': 64,
                        'embedding_dim': 90,
                        'dropout': 0.7,
                        'alphabet_len': len(alphabet),
                        'max_text_len': max_text_len,
                        'num_classes': n_classes}
        lr = 1e-3
        epochs = 25

    else:
        raise ValueError('Wrong model name')

    if epochs == 1:
        logger.warning('Only one epoch!')

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
               noise_level=args.noise_level,
               **model_params)
    logger.info('Total execution time: {:.1f}min'.format((time() - start_time) / 60.))
