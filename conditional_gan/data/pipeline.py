"""
Data pipeline utilities.
"""

import pandas as pd
import torch
import torchvision
from torchvision.transforms import transforms

from data.datasets import ImageDataset


# Construct pytorch dataset and dataloader for the training/testing data within
# an image dataset directory. Also performs all of the standard image dataset
# processing functions (resizing, standardization, etc.).
def process_image_dataset(
        path, image_size=(32, 32), batch_size=64, num_workers=1):
    # read train/test label files to dataframe
    train_df = pd.read_csv('{}train_labels.csv'.format(path))
    test_df = pd.read_csv('{}test_labels.csv'.format(path))

    # convert filename column to absolute paths
    train_df['Filename'] = train_df['Filename'] \
        .map(lambda x: '{}train/{}'.format(path, x))
    test_df['Filename'] = test_df['Filename'] \
        .map(lambda x: '{}test/{}'.format(path, x)).to_list()

    # define the transform chain to process each sample
    # as it is passed to a batch
    #   1. resize the sample (image) to 32x32 (h, w)
    #   2. convert resized sample to Pytorch tensor
    #   3. normalize sample values (pixel values) using
    #      mean 0.5 and stdev 0,5; [0, 255] -> [0, 1]
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    # create train/test datasets
    train_set = ImageDataset(train_df, transform=transform)
    test_set = ImageDataset(test_df, transform=transform)

    # create train/test dataloaders
    train_ldr = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)
    test_ldr = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    return {
        'train_set': train_set,
        'test_set': test_set,
        'train_ldr': train_ldr,
        'test_ldr': test_ldr
    }
