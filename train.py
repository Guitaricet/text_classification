import os
from time import time

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm

import cfg
from text_classification import trainutils, datautils
from text_classification.logger import logger
from text_classification.trainutils import CosineLRWithRestarts


# TODO: add early stopping
# Note: save_model_path and save_results_path are different entities (dir path and file path) better to change this


def train(model,
          train_dataloader,
          val_dataloader,
          noise_level,
          lr=1e-3,
          epochs=10,
          comment='',
          log_every=10,
          save_model_path=None,
          use_annealing=True):
    """
    Train the model, evaluate with different noises

    :param model: torch.nn.module
    :param train_dataloader: noised dataloader
    :param val_dataloader: noised dataloader
    :param noise_level: 0 <= noise_level <= 1, train noise level
    :param lr: learning rate
    :param epochs: number of train epochs
    :param comment: comment for TensorBoard runs name
    :param log_every: log every epochs
    :param save_model_path: path for directory for trained model saving
    :param use_annealing: use CosineLRWithRestarts for lr
    :return: model, results where model is trained model, results is list of dicts
    """
    assert noise_level in cfg.experiment.noise_levels
    if cfg.cuda:
        model.to(torch.device('cuda'))

    start_time = time()

    train_dataloader.__class__.noise_level = noise_level

    model_name = '_{}_lr{}_dropout{}_noise_level{:.4f}'.format(
        model.name, int(-np.log10(lr)), model.dropout_prob, noise_level
    )
    model_name += comment

    writer = SummaryWriter(comment=model_name)
    run_name = list(writer.all_writers.keys())[0]
    logger.info('Writer: %s' % run_name)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    lr_scheduler = CosineLRWithRestarts(optimizer, lr, 1, len(train_dataloader))

    global_step = 0

    loss_f = F.cross_entropy

    for epoch in range(epochs):
        for batch_idx, (text, label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            if use_annealing:
                lr_scheduler.batch_step()

            if cfg.cuda:
                text = text.cuda()
                label = torch.LongTensor(label).cuda()
            else:
                label = torch.LongTensor(label)

            # TODO: change dataloaders and remove premute
            text = text.permute(1, 0, 2)
            prediction = model(text)
            loss = loss_f(prediction, label)

            writer.add_scalar('loss', loss, global_step=global_step)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-1)
            optimizer.step()

            if cfg.cuda:
                torch.cuda.synchronize()

            global_step += 1

        # evaluation
        model.eval()
        train_metrics = trainutils.get_metrics(model, train_dataloader, 0.1)
        val_metrics = trainutils.get_metrics(model, val_dataloader)
        model.train()

        writer.add_scalar('accuracy_train', train_metrics['accuracy'], global_step=global_step)
        writer.add_scalar('f1_train', train_metrics['f1'], global_step=global_step)
        writer.add_scalar('accuracy_val', val_metrics['accuracy'], global_step=global_step)
        writer.add_scalar('f1_val', val_metrics['f1'], global_step=global_step)

        if epoch % log_every == 0 or epoch == epochs-1:
            logger.info('Epoch {}. Global step {}. T={:.2f}min'.format(epoch, global_step, (time() - start_time) / 60.))
            logger.info('In-batch loss      : {:.4f}'.format(float(loss)))
            logger.info('Training accuracy  : {:.4f}, f1: {:.4f}'.format(train_metrics['accuracy'], train_metrics['f1']))
            logger.info('Validation accuracy: {:.4f}, f1: {:.4f}'.format(val_metrics['accuracy'], val_metrics['f1']))

    if save_model_path is not None:
        logger.info('Saving the model')
        filename = '%s.torch' % run_name.split('/')[-1]
        with open(os.path.join(save_model_path, filename), 'wb') as f:
            try:
                torch.save(model, f)
            except Exception as e:
                logger.error(e)
                logger.error('Continuing (likely) without saving')

    return model


def evaluate(model, test_dataloader, noise_levels, evals_per_noise):
    is_training = model.training
    if is_training:
        model.eval()

    results = []

    for _ in range(evals_per_noise):
        for noise_level in noise_levels:
            metrics = trainutils.get_metrics(model, test_dataloader, noise_level)
            results.append(metrics)

    if is_training:
        model.train()

    return results


# def evaluate(model, test_dataloader, noise_level_train, noise_level_test, evals_per_noise_level):
#     model.eval()
#
#     results = []
#     test_metrics = None
#
#     noise_levels = cfg.experiment.noise_levels
#
#     for _ in range(evals_per_noise_level):
#         metrics = trainutils.get_metrics(model, test_noised, noise_level=test_noise)
#
#         if test_noise == noise_level_train:  # train noise level
#             test_metrics = metrics
#
#         results.append(datautils.mk_dataline(
#             model_type=model.name,
#             noise_level_train=noise_level_train,
#             noise_level_test=test_noise,
#             acc_test=metrics['accuracy'],
#             f1_test=metrics['f1'],
#             dropout=dropout,
#             model=model,
#             run_name=run_name
#         ))
#         _maybe_save_results(results, save_results_path)
#
#     metrics = trainutils.get_metrics(model, test_original, noise_level=0)
#     results.append(datautils.mk_dataline(
#         model_type=model.name,
#         epochs=epochs,
#         lr=lr,
#         noise_level_train=noise_level_train,
#         acc_train=train_metrics['accuracy'],
#         f1_train=train_metrics['f1'],
#         noise_level_test=-1,
#         acc_test=metrics['accuracy'],
#         f1_test=metrics['f1'],
#         dropout=dropout,
#         model=model,
#         run_name=run_name
#     ))
#     _maybe_save_results(results, save_results_path)
#
#     logger.info('Original dataset accuracy: {:.4f}, f1: {:.4f}'.format(metrics['accuracy'], metrics['f1']))
#     writer.add_scalar('accuracy_test_original', metrics['accuracy'], global_step=global_step)
#     writer.add_scalar('f1_test_original', metrics['f1'], global_step=global_step)
#
#     logger.info('Final test accuracy:{:.4f}, f1: {:.4f}. Time {} min'
#                 .format(test_metrics['accuracy'], test_metrics['f1'], ((time() - start_time) / 60.)))
#
#
# def _maybe_save_results(results, savepath):
#     if savepath is not None:
#         old_results_df = pd.DataFrame()
#         if os.path.exists(savepath):
#             old_results_df = pd.read_csv(savepath)
#         results_df = pd.DataFrame(results)
#         results_df = pd.concat([old_results_df, results_df], sort=False)
#         results_df.to_csv(savepath)
