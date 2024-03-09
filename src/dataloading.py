"""
Load images from folder, pre-process them and return dataloaders.

@author: Lukas Gandler
"""
import numpy as np
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


def load_OCT(batch_size: int, num_workers: int = 0) -> DataLoader:
    """
    Loads the un-altered OCT images from the OCTID dataset by the university of Waterloo
    available under https://www.openicpsr.org/openicpsr/project/108503/version/V1/view
    :param batch_size:
    :param num_workers:
    :return:
    """

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomCrop((500, 512)),  # Select a random patch within that 500x700 image region
        transforms.Pad([0, 6, 0, 6]),  # Pad to 512x512
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.21074388613859257], std=[0.13993958081559033])
    ])

    dataset = ImageFolder('./datasets/OCTID_IMAGES', transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    return data_loader


def load_processed_OCT(batch_size: int, num_workers: int = 0) -> DataLoader:
    """
    Loads the pre-processed OCT images from the OCTID dataset by the university of Waterloo
    available under https://www.openicpsr.org/openicpsr/project/108503/version/V1/view
    :param batch_size:
    :param num_workers:
    :return:
    """
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomCrop((500, 512)),  # Select a random patch within that 500x700 image region
        transforms.Pad([0, 6, 0, 6]),  # Pad to 512x512
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.21008788933651848], std=[0.1335889231399768])
    ])

    dataset = ImageFolder('./datasets/OCTID_IMAGES_PROCESSED_BIL', transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    return data_loader

# Compute stats about the dataset
# if __name__ == "__main__":
#     mean = 0.0
#     var  = 0.0
#
#     dataloader = load_processed_OCT(batch_size=1)
#     for batch in dataloader:
#         img = batch[0].squeeze()
#         mean += img.mean().item()
#         var  += img.var().item()
#
#     mean /= len(dataloader)
#     var /= len(dataloader)
#
#     std = np.sqrt(var)
#     print(f'Mean: {mean:}, Variance: {var:}, Standard Deviation: {std:}')
#
