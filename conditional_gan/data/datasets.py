"""
torch.utils.data.Dataset classes.
Reference:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

import torch
from PIL import Image


class ImageDataset(torch.utils.data.Dataset):
    """
    Make a PyTorch dataset from a dataframe of image files and labels.
    """

    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # read image and get label
        # NOTE: image must be PIL image for standard PyTorch transforms
        image = Image.open(self.df['Filename'].iloc[idx])
        label = self.df['Label'].iloc[idx]

        # apply any image transform
        if self.transform:
            image = self.transform(image)

        # construct packaged sample
        data = {'image': image, 'label': label}

        return data
