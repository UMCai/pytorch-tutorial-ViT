from torchvision import datasets, models, transforms
import os
import torch
import Config
import torch
from torch.utils.data import Dataset, DataLoader

hymenoptera_data_dir = Config.HYMENOPTERA_DATA_PATH
dogtiny_data_dir = Config.DOGTINY_DATA_PATH
img_size = Config.IMG_SIZE

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def hymenoptera_dataloaders(batch_size,num_workers):
    image_datasets = {x: datasets.ImageFolder(os.path.join(hymenoptera_data_dir, x),
                    data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                    shuffle=True, pin_memory = True, num_workers=num_workers) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(dataset_sizes)
    class_names = image_datasets['train'].classes
    return dataloaders, class_names, dataset_sizes


def dogtiny_dataloaders(batch_size,num_workers):
    train_ds, train_valid_ds = [datasets.ImageFolder(
        os.path.join(dogtiny_data_dir, 'train_valid_test', folder),
        data_transforms[folder]) for folder in ['train', 'val']]
    train_iter, train_valid_iter = [DataLoader(
        dataset, batch_size = batch_size, pin_memory = True, num_workers=num_workers, shuffle=True, drop_last=True)
        for dataset in (train_ds, train_valid_ds)]
    dataloaders = {'train':train_iter, 'val': train_valid_iter}
    # I decide this is not necessary
    #valid_ds, test_ds = [datasets.ImageFolder(
    #    os.path.join(dogtiny_data_dir, 'train_valid_test', folder),
    #    transform=data_transforms) for folder in ['valid', 'test']]
    
    dataset_sizes = {'train': len(train_ds), 'val': len(train_valid_ds)}
    print(dataset_sizes)
    class_names = train_ds.classes
    return dataloaders, class_names, dataset_sizes


