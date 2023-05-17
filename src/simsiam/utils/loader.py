from PIL import ImageFilter
import random

import torch
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_dataloader(cfg, traindir='/home/ueno/bing_al/flowers/train'):
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    train_dataset = datasets.ImageFolder(
        traindir,
        TwoCropsTransform(transforms.Compose(augmentation)))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.train_parameters.batch_size, shuffle=True,
        num_workers=8, pin_memory=True)
    return train_loader, train_dataset

def get_sorted_loader(cfg, traindir='/home/ueno/bing_al/flowers/train'):
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    train_dataset = datasets.ImageFolder(
        traindir,
        TwoCropsTransform(transforms.Compose(augmentation)))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.train_parameters.batch_size, shuffle=False,
        num_workers=1, pin_memory=True)
    return train_loader