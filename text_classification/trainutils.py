import numpy as np

import torch
from torch.nn.data import SubsetRandomSampler


def get_train_valid_loader(dataset, valid_size, batch_size, random_seed=42, shuffle=True, num_workers=4):

    len_dataset = len(dataset)
    indices = list(range(len_dataset))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    val_actual_size = int(len_dataset * valid_size)

    train_idx, valid_idx = indices[:-val_actual_size], indices[-val_actual_size:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=4
    )

    return train_loader, valid_loader


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
