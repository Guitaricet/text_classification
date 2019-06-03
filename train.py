from time import time

import numpy as np

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score


import cfg
from text_classification import trainutils
from text_classification.logger import logger


# Note: save_model_path and save_results_path are different entities (dir path and file path) better to change this


def train(model,
          train_dataloader,
          val_dataloader,
          noise_level,
          lr=cfg.train.lr,
          epochs=cfg.train.epochs,
          comment='',
          use_annealing=True,
          device=cfg.device,
          patience=5):
    """
    Train the model, evaluate with different noises

    :param model: torch.nn.module
    :param train_dataloader: noised dataloader
    :param val_dataloader: noised dataloader
    :param noise_level: 0 <= noise_level <= 1, train noise level
    :param lr: learning rate
    :param epochs: number of train epochs
    :param comment: comment for TensorBoard runs name
    :param save_model_path: path for directory for trained model saving
    :param use_annealing: (deprecated) does not do anything
    :return: model, results where model is trained model, results is list of dicts
    """
    model.to(device)

    start_time = time()

    train_dataloader.dataset.set_noise_level(noise_level)
    val_dataloader.dataset.set_noise_level(noise_level)

    model_name = '{}_noise_level{:.4f}'.format(model.name, noise_level)
    model_name += comment

    saveto = f'models/{model_name}.pt'

    writer = SummaryWriter(comment='_' + model_name)
    run_name = list(writer.all_writers.keys())[0]
    logger.info('Writer: %s' % run_name)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    global_step = 0
    best_metrics = {'f1': 0, 'accuracy': 0}
    impatience = 0
    stop_training = False

    for epoch in range(epochs):
        if stop_training:
            break

        for batch_idx, (text, label) in enumerate(train_dataloader):
            optimizer.zero_grad()

            text = text.to(device)
            label = label.to(device)

            logits = model(text)
            loss = F.cross_entropy(logits, label)

            if batch_idx % 100 == 0:
                writer.add_scalar('loss', loss, global_step=global_step)
                prediction = torch.max(logits, 1).indices

                _label = label.cpu().detach().numpy()
                _prediction = prediction.cpu().detach().numpy()
                train_metrics = {
                    'f1': f1_score(_label, _prediction, average='macro'),
                    'accuracy': accuracy_score(_label, _prediction)
                }
                writer.add_scalar('accuracy/train', train_metrics['accuracy'],
                                  global_step=global_step)
                writer.add_scalar('f1/train', train_metrics['f1'], global_step=global_step)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            if batch_idx % 1000 == 0:
                model.eval()
                val_metrics = trainutils.get_metrics(model, val_dataloader, frac=0.25)
                model.train()
                writer.add_scalar('accuracy/val', val_metrics['accuracy'], global_step=global_step)
                writer.add_scalar('f1/val', val_metrics['f1'], global_step=global_step)

                lr_scheduler.step(val_metrics['f1'])

                if val_metrics['f1'] < best_metrics['f1']:
                    impatience += 1
                    if impatience > patience and epoch > 0:
                        stop_training = True  # noqa E701
                        break
                else:
                    impatience = 0
                    best_metrics = val_metrics
                    torch.save(model.state_dict(), saveto)

            # torch.cuda.synchronize()

            global_step += 1

        if epoch % 5 == 0 or epoch == epochs-1 or stop_training:
            logger.info('Epoch {}. Global step {}. T={:.2f}min'.format(epoch, global_step, (time() - start_time) / 60.))
            logger.info('In-batch loss      : {:.4f}'.format(float(loss)))
            logger.info('Training accuracy  : {:.4f}, f1: {:.4f}'.format(train_metrics['accuracy'], train_metrics['f1']))  # noqa E501
            logger.info('Validation accuracy: {:.4f}, f1: {:.4f}'.format(val_metrics['accuracy'], val_metrics['f1']))

    model.load_state_dict(torch.load(saveto))

    return model


# TODO: move to trainutils?
def evaluate_on_noise(model, test_dataloader, noise_levels, evals_per_noise, verbose=True):
    is_training = model.training
    if is_training:
        model.eval()

    results = []

    for _ in range(evals_per_noise):
        for noise_level in noise_levels:
            metrics = trainutils.get_metrics(model, test_dataloader, noise_level=noise_level)
            metrics = {'noise_level_test': noise_level,
                       'acc_test': metrics['accuracy'],
                       'f1_test': metrics['f1']}
            if verbose:
                logger.info('Test     accuracy  : {:.4f}, f1: {:.4f}'.format(metrics['acc_test'], metrics['f1_test']))
            results.append(metrics)

    if is_training:
        model.train()

    return results
