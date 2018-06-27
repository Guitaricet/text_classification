import logging
import argparse

from time import time
from random import random, choice

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import torch
import nltk
import torchtext

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn import init
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

from pymystem3 import Mystem
from nltk.tokenize import word_tokenize
from tensorboardX import SummaryWriter
from tqdm import tqdm as tqdm

from text_classification.trainutils import get_metrics, get_train_valid_loader
from text_classification.datautils import HieracialIMDB, HieracialMokoron, mk_dataline
from text_classification.layers import AttentionedYoonKimModel

nltk.download('punkt')

parser = argparse.ArgumentParser()
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--aijun', default=False, action='store_true')
parser.add_argument('--madrugado', default=False, action='store_true')
parser.add_argument('--num-workers', type=int, default=1)
args = parser.parse_args()

assert args.aijun ^ args.madrugado, '--aijun or --madrugado should be specified'

logger = logging.getLogger()

fileHandler = logging.FileHandler('main.log')
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
logger.addHandler(consoleHandler)

logger.setLevel('DEBUG')

time_total = time()

np.random.seed(42)
CUDA = torch.cuda.is_available()

BATCH_SIZE = 32
VALID_SIZE = 0.1

NOISE_LEVELS = [0, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]

MAX_WORD_LEN = 8
# MAX_TEXT_LEN = 32
MAX_TEXT_LEN = 128

ALPHABET = ['<UNK>'] + ['\n'] + [s for s in """ 0123456789-,;.!?:'’’/\|_@#$%ˆ&* ̃‘+-=<>()[]{}"""]
# if lang.upper() in ('RU', 'RUS'):
#     ALPHABET += [s for s in 'абвгдеёжзийклмнопрстуфхцчщъыьэюя']
ALPHABET += [s for s in 'abcdefghijklmnopqrstuvwxyz']

# if noemoji:
#     ALPHABET = [c for c in ALPHABET if c not in ('(', ')')]  # выключаем эмоджи

ALPHABET_LEN = len(ALPHABET)
char2int = {s: i for s, i in zip(ALPHABET, range(ALPHABET_LEN))}

# TODO: move this to config
if args.madrugado:
    basepath = '../data/IMDB/splits/'
if args.aijun:
    basepath = '/media/data/nlp/sentiment/IMDB/splits/'


def run_model_with(noise_level, n_filters, cnn_kernel_size, hidden_dim_out, dropout=0.5,
                   lr=1e-4, epochs=30, heads=1, comment='_test', log_every=10, init_function=None, _model=None):
    start_time = time()
#     HieracialIMDB.noise_level = noise_level
#     task='IMDB binary classification'
    HieracialMokoron.noise_level = noise_level
    task='IMDB binary classification'

    if _model is None:
        model = AttentionedYoonKimModel(
            n_filters=n_filters, cnn_kernel_size=cnn_kernel_size, hidden_dim_out=hidden_dim_out, dropout=dropout,
            init_function=init_function, heads=heads
        )
        if CUDA:
            model.cuda()
        model.train()

    else:
        model = _model

    model_name = '_AttentionedYoonKim_lr%s_dropout%s_noise_level%s_spacy_wordlen8_heads%s' % (
        int(-np.log10(lr)), model.dropout, noise_level, model.heads
    ) + comment
    
    if '(' not in ALPHABET:
        model_name += '_no_emoji'

    writer = SummaryWriter(comment=model_name)
    if len(list(writer.all_writers.keys())) > 1:
        logger.info('More than one writer! 0_o')
        logger.info(list(writer.all_writers.keys()))

    run_name = list(writer.all_writers.keys())[0]
    logger.info('Writer: %s' % run_name)

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    optimizer.zero_grad()
    
    global_step = 0

    loss_f = F.cross_entropy

    for epoch in range(epochs):

        for batch_idx, (text, label) in enumerate(dataloader):
            optimizer.zero_grad()

            if CUDA:
                text = Variable(text.cuda())
                label = Variable(torch.LongTensor(label).cuda())
            else:
                text = Variable(text)
                label = Variable(torch.LongTensor(label))

            text = text.permute(1, 0, 2)
            prediction = model(text)
            loss = loss_f(prediction, label)

            writer.add_scalar('loss', loss.data[0], global_step=global_step)

            loss.backward()        
            torch.nn.utils.clip_grad_norm(model.parameters(), 1e-1)
            optimizer.step()

            if CUDA:
                torch.cuda.synchronize()
            global_step += 1

        # evaluation
        if epoch % log_every == 0:
            logger.info('Epoch %s. Global step %s. T=%s min' % (epoch, global_step, (time() - start_time) / 60.))
            logger.info('Loss               : %s' % loss.data[0])

        # in-batch
        _, idx = torch.max(prediction, 1)
        _labels = label.data.tolist()
        _predictions = idx.data.tolist()
        acc = accuracy_score(_labels, _predictions)
        f1 = f1_score(_labels, _predictions)
        writer.add_scalar('accuracy_train', acc, global_step=global_step)
        writer.add_scalar('f1_train', f1, global_step=global_step)
        if epoch % log_every == 0:
            logger.info('In-batch accuracy  : %s', acc)

        # validation
        metrics = get_metrics(model, val_dataloader)
        if epoch % log_every == 0:
            logger.info('Validation accuracy: %s, f1: %s' % (metrics['accuracy'], metrics['f1']))
            logger.info('\n')

        writer.add_scalar('accuracy_val', metrics['accuracy'], global_step=global_step)
        writer.add_scalar('f1_val', metrics['f1'], global_step=global_step)

    with open('models/%s.torch' % run_name.split('/')[-1], 'wb') as f:
        try:
            torch.save(model, f)
        except Exception as e:
            logger.error(e)
            logger.error('Continuing (probably) without saving')


    logger.info('Calculating validation metrics... Time %s min' % ((time() - start_time) / 60.))
    metrics_train = get_metrics(model, dataloader)
    acc_train = metrics_train['accuracy']
    f1_train = metrics_train['f1']

    for test_noise in tqdm(NOISE_LEVELS, leave=False):
        metrics = get_metrics(model, test, test_noise)
        if test_noise == noise_level:
            metrics_test = metrics

        acc_test = metrics['accuracy']
        f1_test = metrics['f1']
        results.append(mk_dataline(
            model_type='charCNN', epochs=epochs, lr=lr,
            noise_level_train=noise_level, acc_train=acc_train, f1_train=f1_train,
            noise_level_test=test_noise, acc_test=acc_test, f1_test=f1_test,
            dropout=dropout, model=model,
            init_function=init_function,
            run_name=run_name,
            task=task
        ))

    # test original
    metrics = get_metrics(model, test_original)
    results.append(mk_dataline(
        model_type='charCNN', epochs=epochs, lr=lr,
        noise_level_train=noise_level, acc_train=acc_train, f1_train=f1_train,
        noise_level_test=-1, acc_test=metrics['accuracy'], f1_test=metrics['f1'],
        dropout=dropout, model=model,
        init_function=init_function,
        run_name=run_name,
        task=task
    ))
    
    logger.info('Original dataset: acc %s, f1 %s' % (metrics['accuracy'], metrics['f1']))
    writer.add_scalar('accuracy_test_original', metrics['accuracy'], global_step=global_step)
    writer.add_scalar('f1_test_original', metrics['f1'], global_step=global_step)

    logger.info('Final test metrics: %s, Time %s min' % (metrics_test, ((time() - start_time) / 60.)))
    if metrics_test is not None:
        writer.add_scalar('accuracy_test_final', metrics_test['accuracy'], global_step=global_step)
        writer.add_scalar('f1_test_final', metrics_test['f1'], global_step=global_step)
    logger.info('\n')
    # model is in EVAL mode!
    return model

if __name__ == '__main__':
    logger.info('Script is started')
    train = HieracialMokoron(basepath + 'train.csv', 'text_spellchecked')
    valid = HieracialMokoron(basepath + 'validation.csv', 'text_spellchecked')
    test = HieracialMokoron(basepath + 'test.csv', 'text_spellchecked')

    test_original = HieracialMokoron(basepath + 'test.csv', 'text_original')

    dataloader = torch.utils.data.DataLoader(train, BATCH_SIZE, shuffle=True, num_workers=args.num_workers)
    val_dataloader = torch.utils.data.DataLoader(valid, BATCH_SIZE, shuffle=True, num_workers=args.num_workers)

    results = []

    # logger.info('Testing the script...')
    # logger.info('Running one epoch and evaluation to check evetything is ok')
    # logger.info('...')
    # logger.info("At least to check it won't crush")

    # run_model_with(
    #     noise_level=0, n_filters=256, cnn_kernel_size=5, hidden_dim_out=128, dropout=0.5,
    #     lr=1e-3, epochs=1, heads=1, comment='_test'
    # )
    # pd.DataFrame(results).to_csv('results/AttentionedYoonKim_mokoron_test.csv')

    if not args.test:
        logger.info('Models with one attention head')
        for noise_level in tqdm(NOISE_LEVELS[5:], leave=False):
            run_model_with(
                noise_level=noise_level, n_filters=256, cnn_kernel_size=5, hidden_dim_out=128, dropout=0.5,
                lr=1e-3, epochs=30, heads=1, comment='_IMDB'
            )
        logger.info('Saving results table')
        filename1 = 'results/AttentionedYoonKim_IMDB_heads1_last5.csv'
        pd.DataFrame(results).to_csv(filename1)
        logger.info('Saved with name %s' % filename1)

        # logger.info('Models with two attention heads')
        # for noise_level in tqdm(NOISE_LEVELS, leave=False):
        #     run_model_with(
        #         noise_level=noise_level, n_filters=256, cnn_kernel_size=5, hidden_dim_out=128, dropout=0.5, lr=1e-3, epochs=30, heads=2
        #     )
        # logger.info('Saving results table')
        # filename2 = 'results/AttentionedYoonKim_mokoron_heads2.csv'
        # pd.DataFrame(results).to_csv(filename2)
        # logger.info('Saved with name %s' % filename2)

        # logger.info('Models with four attention heads')
        # for noise_level in tqdm(NOISE_LEVELS, leave=False):
        #     run_model_with(
        #         noise_level=noise_level, n_filters=256, cnn_kernel_size=5, hidden_dim_out=128, dropout=0.5, lr=1e-3, epochs=30, heads=4
        #     )
        # logger.info('Saving results table')
        # filename4 = 'results/AttentionedYoonKim_mokoron_heads4.csv'
        # pd.DataFrame(results).to_csv(filename4)
        # logger.info('Saved with name %s' % filename4)

    logger.info('Success!')
    logger.info('Total execution time: %smin' % ((time() - time_total) // 60))
