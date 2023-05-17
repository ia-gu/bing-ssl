import torch
from torchvision import transforms, datasets
from torch.utils.data import ConcatDataset

def get_data(data_cfg):
    model_name = 'resnet'
    channels = 3
    if data_cfg.name=='flowers':
        classes = [0]*102
        train_data_path = 'flowers/train'
        test_data_path = 'flowers/test'
        train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.ImageFolder(root=train_data_path, transform=train_transform)
        test_dataset = datasets.ImageFolder(root=test_data_path, transform=test_transform)
    
    if data_cfg.name=='pets':
        classes = [0]*37
        train_data_path = 'pets/train'
        test_data_path = 'pets/test'
        train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.ImageFolder(root=train_data_path, transform=train_transform)
        test_dataset = datasets.ImageFolder(root=test_data_path, transform=test_transform)
        
    if data_cfg.name=='foods':
        classes = [0]*101
        train_data_path = 'flowers/train'
        test_data_path = 'flowers/test'
        train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.ImageFolder(root=train_data_path, transform=train_transform)
        test_dataset = datasets.ImageFolder(root=test_data_path, transform=test_transform)

    return train_dataset, test_dataset, model_name, channels, classes