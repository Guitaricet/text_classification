import numpy as np

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score

import cfg
from text_classification.logger import logger

from allennlp.modules.elmo import batch_to_ids


class CosineLRWithRestarts:
    def __init__(self, optimizer, max_lr, cycle_len, n_batches):
        """
        :param max_lr: max lr rate
        :param cycle_len: cycle len in epochs
        :param n_batches: length of epoch in batches
        """
        self._optimizer = optimizer
        self.max_lr = max_lr
        self.cycle_len = cycle_len
        self.n_batches = n_batches
        self._period = self.n_batches * self.cycle_len
        self._cycle_iter = 0
        self._cycle_number = 0

    def __next__(self):
        cos_out = np.cos(np.pi * self._cycle_iter / self._period) + 1
        self._cycle_iter += 1
        if self._cycle_iter == self._period:
            self._cycle_iter = 0
            self._cycle_number += 1
        return self.max_lr / 2 * cos_out

    def batch_step(self):
        """
        Updates lr of optimizer
        """
        lr = self.__next__()
        for param in self._optimizer.param_groups:
            param['lr'] = lr


def get_dataloaders(dataset,
                    testset,
                    batch_size,
                    valid_size=None,
                    random_seed=42,
                    shuffle=True,
                    num_workers=cfg.train.num_workers,
                    validset=None):
    """
    Split dataset into train and valid and make dataloaders

    Only one of valid_size or validset should be specified
    Test dataloader also made here for common dataloaders structure
    Test dataloader is not shuffled
    :param validset: if specified
    :param dataset: torch.dataset, will be separated into train and validation sets
    :param testset: torch.dataset
    :param valid_size: 0 < valid_size < 1
    :param batch_size: int, batch size
    :param random_seed: random seed for
    :param shuffle: shuffle dataset before split
    :param num_workers: number of CPU workers for each dataloader
    :return: train dataloader, validation dataloader
    """
    assert (validset is not None) ^ (valid_size is not None), 'Only one of valid_size or validset should be specified'

    if valid_size is not None:
        len_dataset = len(dataset)
        indices = list(range(len_dataset))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        val_actual_size = int(len_dataset * valid_size)

        train_idx, valid_idx = indices[:-val_actual_size], indices[-val_actual_size:]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
        )
        valid_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers
        )
    else:
        train_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
        valid_loader = DataLoader(
            validset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

    test_loader = DataLoader(
        testset, batch_size=batch_size, num_workers=num_workers
    )

    return train_loader, valid_loader, test_loader


def get_metrics(model, test_data, noise_level=None, frac=1.0):
    """
    Evaluate the model

    :param model: torch module
    :param test_data: torch Dataset or DataLoader
    :param noise_level: 0 <= noise_level <= 1
    :param frac: 0 < frac <=1, which part of test_data to use for evaluation
    """
    is_training_mode = model.training
    if is_training_mode:
        logger.warning('Model is evaluating in training mode!')
        model.eval()
        logger.info('Set the model into eval mode')

    if isinstance(test_data, torch.utils.data.Dataset):
        assert False, 'Do not use '
        if noise_level is not None:
            test_data.noise_level = noise_level

        test_dataloader = DataLoader(
            test_data, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers
        )
    else:
        assert isinstance(test_data, torch.utils.data.DataLoader)
        test_dataloader = test_data

    if noise_level is not None:
        prev_noise_level = test_dataloader.dataset.noise_level
        test_dataloader.dataset.noise_level = noise_level

    predictions = []
    labels = []

    with torch.no_grad():
        data_length = len(test_dataloader)

        for i, (text, label) in enumerate(test_dataloader):
            if i >= frac * data_length:
                break
            if cfg.cuda:
                if cfg.elmo:
                    text = batch_to_ids(text)
                text = text.cuda()

            text = text.permute(1, 0, 2)

            prediction = model(text)
            _, idx = torch.max(prediction, 1)

            # if i == 0:
            #     _, symb = text[9]
            #     logger.info('Texts ninth symbols:\n %s' % symb)
            #     logger.info('Prediction:\n %s' % prediction)

            predictions.extend(idx.tolist())
            labels.extend(label.tolist())

        # logger.info(labels)
        # logger.info(predictions)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')

    if is_training_mode:
        model.train()

    if noise_level is not None:
        test_dataloader.dataset.noise_level = prev_noise_level

    return {'accuracy': acc, 'f1': f1}
