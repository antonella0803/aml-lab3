import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_dataloaders(data_dir='dataset/tiny-imagenet-200', batch_size=64, num_workers=2):
    """
    Crea e restituisce i DataLoader di Train e Validation per Tiny-ImageNet.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')

    image_datasets = {
        'train': datasets.ImageFolder(train_path, data_transforms['train']),
        'val': datasets.ImageFolder(val_path, data_transforms['val'])
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    }

    return dataloaders, image_datasets['train'].classes
