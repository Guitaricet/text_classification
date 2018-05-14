import logging

from time import time
from random import random, choice

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn import init
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

import torchtext

from tensorboardX import SummaryWriter
from tqdm import tqdm as tqdm
from pymystem3 import Mystem

parser = argparse.ArgumentParser()
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--aijun', default=False, action='store_true')
parser.add_argument('--madrugado', type=False, default='store_true')


rootLogger = logging.getLogger()

fileHandler = logging.FileHandler('main.log')
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
rootLogger.addHandler(consoleHandler)

time_total = time()

np.random.seed(42)
CUDA = torch.cuda.is_available()

BATCH_SIZE = 32
VALID_SIZE = 0.1

NOISE_LEVELS = [0, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]


MAX_WORD_LEN = 8
MAX_TEXT_LEN = 32

ALPHABET = ['<UNK>'] + ['\n'] + [s for s in """ 0123456789-,;.!?:'’’/\|_@#$%ˆ&* ̃‘+-=<>()[]{}"""]
ALPHABET += [s for s in 'абвгдеёжзийклмнопрстуфхцчщъыьэюя']
ALPHABET += [s for s in 'abcdefghijklmnopqrstuvwxyz']

ALPHABET = [c for c in ALPHABET if c not in ('(', ')')]  # выключаем эмоджи

ALPHABET_LEN = len(ALPHABET)
char2int = {s: i for s, i in zip(ALPHABET, range(ALPHABET_LEN))}


class HieracialMokoron(torch.utils.data.Dataset):
    """
    Zero vector for padding.
    """
    noise_level = 0

    def __init__(self, filepath, text_field, maxwordlen=MAX_WORD_LEN, maxtextlen=MAX_TEXT_LEN):
        self.alphabet = ALPHABET

        self.mystem = Mystem()
        self.text_field = text_field
        self.data = pd.read_csv(filepath)
        self.maxwordlen = maxwordlen
        self.maxtextlen = maxtextlen
        self.char2int = {s: i for s, i in zip(self.alphabet, range(len(self.alphabet)))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data.iloc[idx]
        text = line[self.text_field].lower()
        label = int(line.sentiment == 1.)

        if self.noise_level > 0:
            text = self._noise_generator(text)

        text = self._tokenize(text)
        text = self._preprocess(text)
        return text, label

    def _tokenize(self, text):
        return [res['text'] for res in self.mystem.analyze(text) if res['text'] != ' ']

    def _noise_generator(self, string):
        noised = ""
        for c in string:
            if random() > self.noise_level:
                noised += c
            if random() < self.noise_level:
                noised += choice(self.alphabet)
        return noised

    def _one_hot(self, char):
        zeros = np.zeros(len(self.alphabet))
        if char in self.char2int:
            zeros[self.char2int[char]] = 1.
        else:
            zeros[self.char2int['<UNK>']] = 1.

    def _preprocess(self, text):
        _text_tensor = torch.zeros([self.maxwordlen * self.maxtextlen, len(self.alphabet)])
        
        for i, token in enumerate(text):
            if i >= self.maxtextlen:
                break
            for j, char in enumerate(token):
                if j >= self.maxwordlen:
                    break
                _text_tensor[i*self.maxwordlen + j, char2int.get(char, char2int['<UNK>'])] = 1.

        return _text_tensor

    def onehot2text(one_hotted_text, batch_size=None, show_pad=False):
        if batch_size is None:
            text = ''
            max_values, idx = torch.max(one_hotted_text, 1)
            for c, i in enumerate(idx):
                if max_values[c] == 0:
                    if show_pad:
                        symb = '<PAD>'
                    else:
                        symb = ''
                else:
                    symb = ALPHABET[i]
                text += symb
            return text
        else:
            texts = []
            for text in one_hotted_text:
                texts.append(onehot2text(one_hotted_text, batch_size=None))
            return texts

def onehot2text(one_hotted_text, batch_size=None, show_pad=False):
    if batch_size is None:
        text = ''
        max_values, idx = torch.max(one_hotted_text, 1)
        for c, i in enumerate(idx):
            if max_values[c] == 0:
                if show_pad:
                    symb = '<PAD>'
                else:
                    symb = ''
            else:
                symb = ALPHABET[i]
            text += symb
        return text
    else:
        texts = []
        for text in one_hotted_text:
            texts.append(onehot2text(one_hotted_text, batch_size=None))
        return texts

def get_metrics(model, test_data, noise_level=None):
    """
    :param test_data: dataset or dataloader

    Moder will be in TRAIN mode after that
    """
    model.eval()

    predictions = []
    lables = []
    
    if isinstance(test_data, torch.utils.data.Dataset):
        if noise_level is not None:
            test_data.noise_level = noise_level

        test_dataloader = torch.utils.data.DataLoader(
            test_data, batch_size=BATCH_SIZE
        )
    else:
        assert isinstance(test_data, torch.utils.data.DataLoader)
        test_dataloader = test_data

    for text, label in test_dataloader:
        if CUDA:
            text = Variable(text.cuda())
        else:
            text = Variable(text)

        text = text.permute(1, 0, 2)  # (1, 0, 2) for RNN
        prediction = model(text)

        _, idx = torch.max(prediction, 1)
        predictions += idx.data.tolist()
        lables += label.tolist()

    acc = accuracy_score(lables, predictions)
    f1 = f1_score(lables, predictions)
    model.train()
    return {'accuracy': acc, 'f1': f1}

if args.madrugado:
    basepath = '../data/ru-mokoron/splits/'
if args.aijun:
    basepath = '/media/data/nlp/sentiment/ru-mokoron/splits/'


train = HieracialMokoron(basepath + 'train.csv', 'text_spellchecked')
valid = HieracialMokoron(basepath + 'validation.csv', 'text_spellchecked')
test = HieracialMokoron(basepath + 'test.csv', 'text_spellchecked')

test_original = HieracialMokoron(basepath + 'test.csv', 'text_original')

dataloader = torch.utils.data.DataLoader(train, BATCH_SIZE, shuffle=True, num_workers=4)
val_dataloader = torch.utils.data.DataLoader(valid, BATCH_SIZE, shuffle=True, num_workers=4)

results = []

# https://github.com/akurniawan/pytorch-transformer
class MultiHeadAttention(nn.Module):
    def __init__(self,
                 query_dim,
                 key_dim,
                 num_units,
                 dropout_p=0.5,
                 h=8,
                 is_masked=False):
        super(MultiHeadAttention, self).__init__()

        if query_dim != key_dim:
            raise ValueError("query_dim and key_dim must be the same")
        if num_units % h != 0:
            raise ValueError("num_units must be dividable by h")
        if query_dim != num_units:
            raise ValueError("to employ residual connection, the number of "
                             "query_dim and num_units must be the same")

        self._num_units = num_units
        self._h = h
        self._key_dim = Variable(torch.FloatTensor([key_dim]))
        if CUDA:
            self._key_dim = self._key_dim.cuda()
        self._dropout_p = dropout_p
        self._is_masked = is_masked

        self.query_layer = nn.Linear(query_dim, num_units, bias=False)
        self.key_layer = nn.Linear(key_dim, num_units, bias=False)
        self.value_layer = nn.Linear(key_dim, num_units, bias=False)
        self.bn = nn.BatchNorm1d(num_units)

    def forward(self, query, keys):
        Q = self.query_layer(query)
        K = self.key_layer(keys)
        V = self.value_layer(keys)

        # split each Q, K and V into h different values from dim 2
        # and then merge them back together in dim 0
        chunk_size = int(self._num_units / self._h)
        Q = torch.cat(Q.split(split_size=chunk_size, dim=2), dim=0)
        K = torch.cat(K.split(split_size=chunk_size, dim=2), dim=0)
        V = torch.cat(V.split(split_size=chunk_size, dim=2), dim=0)

        # calculate QK^T
        attention = torch.matmul(Q, K.transpose(1, 2))
        # normalize with sqrt(dk)
        attention = attention / torch.sqrt(self._key_dim)
        # use masking (usually for decoder) to prevent leftward
        # information flow and retains auto-regressive property
        # as said in the paper
        if self._is_masked:
            diag_vals = attention[0].sign().abs()
            diag_mat = diag_vals.tril()
            diag_mat = diag_mat.unsqueeze(0).expand(attention.size())
            # we need to enforce converting mask to Variable, since
            # in pytorch we can't do operation between Tensor and
            # Variable
            mask = Variable(
                torch.ones(diag_mat.size()) * (-2**32 + 1), requires_grad=False)
            # this is some trick that I use to combine the lower diagonal
            # matrix and its masking. (diag_mat-1).abs() will reverse the value
            # inside diag_mat, from 0 to 1 and 1 to zero. with this
            # we don't need loop operation andn could perform our calculation
            # faster
            attention = (attention * diag_mat) + (mask * (diag_mat-1).abs())
        # put it to softmax
        attention = F.softmax(attention, dim=-1)
        # apply dropout
        attention = F.dropout(attention, self._dropout_p)
        # multiplyt it with V
        attention = torch.matmul(attention, V)
        # convert attention back to its input original size
        restore_chunk_size = int(attention.size(0) / self._h)
        attention = torch.cat(
            attention.split(split_size=restore_chunk_size, dim=0), dim=2)
        # residual connection
        attention += query
        # apply batch normalization
#         attention = self.bn(attention.transpose(1, 2)).transpose(1, 2)

        return attention

class AttentionedYoonKimModel(nn.Module):
    def __init__(self,
                 n_filters,
                 cnn_kernel_size,
                 hidden_dim_out,
                 dropout=0.5,
                 init_function=None,
                 embedding_dim=len(ALPHABET),
                 pool_kernel_size=MAX_WORD_LEN,
                 heads=1):
        """
        CharCNN-WordRNN model with multi-head attention
        Default pooling is MaxOverTime pooling
        """
        assert cnn_kernel_size % 2  # for 'same' padding

        super(AttentionedYoonKimModel, self).__init__()
        self.dropout = dropout
        self.init_function = init_function
        self.embedding_dim = embedding_dim
        self.n_filters = n_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.hidden_dim_out = hidden_dim_out
        self.heads = heads

        self.embedding = nn.Linear(len(ALPHABET), embedding_dim)
        self.chars_cnn = nn.Sequential(
            nn.Conv1d(embedding_dim, n_filters, kernel_size=cnn_kernel_size, stride=1, padding=int(cnn_kernel_size - 1) // 2),  # 'same' padding
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel_size)
        )
        if init_function is not None:
            self.chars_cnn[0].weight = init_function(self.chars_cnn[0].weight)

        _conv_stride = 1  # by default
        _pool_stride = pool_kernel_size  # by default
        # I am not sure this formula is always correct:
        self.conv_dim = n_filters * max(1, int(((MAX_WORD_LEN - cnn_kernel_size) / _conv_stride - pool_kernel_size) / _pool_stride + 1))

        self.words_rnn = nn.GRU(self.conv_dim, hidden_dim_out, dropout=dropout)
        self.attention = MultiHeadAttention(hidden_dim_out, hidden_dim_out, hidden_dim_out, dropout_p=self.dropout, h=self.heads)
        self.projector = nn.Linear(hidden_dim_out, 2)

    def forward(self, x):
        batch_size = x.size(1)
        # TODO: hadrcode! (for CUDA)
        words_tensor = Variable(torch.zeros(MAX_TEXT_LEN, batch_size, self.conv_dim)).cuda()

        for i in range(MAX_TEXT_LEN):
            word = x[i * MAX_WORD_LEN : (i + 1) * MAX_WORD_LEN, :]
            word = self.embedding(word)
            word = word.permute(1, 2, 0)
            word = self.chars_cnn(word)
            word = word.view(word.size(0), -1)
            words_tensor[i, :] = word

        x, _ = self.words_rnn(words_tensor)
        x = self.attention(x, x)
        x = self.projector(x[-1])
        return x

def model_params_num(model):
    return sum(np.prod(list(p.size())) for p in model.parameters())

def mk_dataline(model_type, epochs, lr, noise_level_train, noise_level_test, acc_train, acc_test,
                f1_train, f1_test, dropout, model, run_name, task, init_function=None):
    return {
        'task': task,
        'model_type': model_type,
        'trainable_params': model_params_num(model), 'dropout': dropout, 'init_function': init_function,
        'epochs': epochs, 'lr': lr,
        'noise_level_train': noise_level_train, 'noise_level_test': noise_level_test,
        'acc_train': acc_train, 'acc_test': acc_test,
        'f1_train': f1_train, 'f1_test': f1_test,
        'model_desc': str(model),
        'run_name': run_name,
        'data_desc': 'MaxWordLen %s, MaxTexLen %s' % (MAX_WORD_LEN, MAX_TEXT_LEN)
    }

def run_model_with(noise_level, n_filters, cnn_kernel_size, hidden_dim_out, dropout=0.5,
                   lr=1e-4, epochs=30, heads=1, comment='_test', log_every=10, init_function=None, _model=None):
    start_time = time()
#     HieracialIMDB.noise_level = noise_level
#     task='IMDB binary classification'
    HieracialMokoron.noise_level = noise_level
    task='Mokoron binary classification'

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
        logging.info('More than one writer! 0_o')
        logging.info(list(writer.all_writers.keys()))

    run_name = list(writer.all_writers.keys())[0]
    logging.info('Writer: %s' % run_name)

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
            logging.info('Epoch %s. Global step %s. T=%s min' % (epoch, global_step, (time() - start_time) / 60.))
            logging.info('Loss               : %s' % loss.data[0])

        # in-batch
        _, idx = torch.max(prediction, 1)
        _labels = label.data.tolist()
        _predictions = idx.data.tolist()
        acc = accuracy_score(_labels, _predictions)
        f1 = f1_score(_labels, _predictions)
        writer.add_scalar('accuracy_train', acc, global_step=global_step)
        writer.add_scalar('f1_train', f1, global_step=global_step)
        if epoch % log_every == 0:
            logging.info('In-batch accuracy  :', acc)

        # validation
        metrics = get_metrics(model, val_dataloader)
        if epoch % log_every == 0:
            logging.info('Validation accuracy: %s, f1: %s' % (metrics['accuracy'], metrics['f1']))
            logging.info('\n')

        writer.add_scalar('accuracy_val', metrics['accuracy'], global_step=global_step)
        writer.add_scalar('f1_val', metrics['f1'], global_step=global_step)

    with open('models/%s.torch' % run_name.split('/')[-1], 'wb') as f:
        try:
            torch.save(model, f)
        except Exception as e:
            logging.error(e)
            logging.error('Continuing (probably) without saving')

        
    logging.info('Calculating validation metrics... Time %s min' % ((time() - start_time) / 60.))
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
    
    logging.info('Original dataset: acc %s, f1 %s' % (metrics['accuracy'], metrics['f1']))
    writer.add_scalar('accuracy_test_original', metrics['accuracy'], global_step=global_step)
    writer.add_scalar('f1_test_original', metrics['f1'], global_step=global_step)

    logging.info('Final test metrics: %s, Time %s min' % (metrics_test, ((time() - start_time) / 60.)))
    if metrics_test is not None:
        writer.add_scalar('accuracy_test_final', metrics_test['accuracy'], global_step=global_step)
        writer.add_scalar('f1_test_final', metrics_test['f1'], global_step=global_step)
    logging.info('\n')
    # model is in EVAL mode!
    return model

if __name__ == 'main':

    logging.info('Testing the script...')
    logging.info('Running one epoch and evaluation to check evetything is ok')
    logging.info('...')
    logging.info("At least to check it won't crush")

    results = []
    run_model_with(
        noise_level=0, n_filters=256, cnn_kernel_size=5, hidden_dim_out=128, dropout=0.5,
        lr=1e-3, epochs=1, heads=1, comment='_test'
    )
    pd.DataFrame(results).to_csv('results/AttentionedYoonKim_mokoron_test.csv')

    if not args.test:
        logging.info('Models with one attention head')
        for noise_level in tqdm(NOISE_LEVELS, leave=False):
            run_model_with(
                noise_level=noise_level, n_filters=256, cnn_kernel_size=5, hidden_dim_out=128, dropout=0.5,
                lr=1e-3, epochs=30, heads=1, comment='_mokoron'
            )
        logging.info('Saving results table')
        filename1 = 'results/AttentionedYoonKim_mokoron_heads1.csv'
        pd.DataFrame(results).to_csv(filename1)
        logging.info('Saved with name %s' % filename1)

        logging.info('Models with two attention heads')
        for noise_level in tqdm(NOISE_LEVELS, leave=False):
            run_model_with(
                noise_level=noise_level, n_filters=256, cnn_kernel_size=5, hidden_dim_out=128, dropout=0.5, lr=1e-3, epochs=30, heads=2
            )
        logging.info('Saving results table')
        filename2 = 'results/AttentionedYoonKim_mokoron_heads2.csv'
        pd.DataFrame(results).to_csv(filename2)
        logging.info('Saved with name %s' % filename2)

        logging.info('Models with four attention heads')
        for noise_level in tqdm(NOISE_LEVELS, leave=False):
            run_model_with(
                noise_level=noise_level, n_filters=256, cnn_kernel_size=5, hidden_dim_out=128, dropout=0.5, lr=1e-3, epochs=30, heads=4
            )
        logging.info('Saving results table')
        filename4 = 'results/AttentionedYoonKim_mokoron_heads4.csv'
        pd.DataFrame(results).to_csv(filename4)
        logging.info('Saved with name %s' % filename4)

    logging.info('Success!')
    logging.info('Total execution time: %smin' % ((time() - time_total) // 60))
