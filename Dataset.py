from torchvision import datasets, models, transforms
import os
import torch
import Config
import torch
from torch.utils.data import Dataset

hymenoptera_data_dir = Config.DATA_PATH
batch_size = Config.BATCH_SIZE
img_size = Config.IMG_SIZE
num_workers = Config.NUM_WORKERS

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
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




def hymenoptera_dataloaders():
    image_datasets = {x: datasets.ImageFolder(os.path.join(hymenoptera_data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, pin_memory = True, num_workers=num_workers)
                 for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(dataset_sizes)
    class_names = image_datasets['train'].classes
    return dataloaders, class_names, dataset_sizes

