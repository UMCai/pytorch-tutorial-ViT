import Config
from Dataset import hymenoptera_dataloaders
from Trainer import train_classification_model, train_classification_model_ignite
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
#from torchsummary import summary

cudnn.benchmark = True
device = Config.DEVICE
model = Config.MODEL
num_epochs = Config.NUM_EPOCHS
lr = Config.LR

if __name__ == '__main__':
    
    data = hymenoptera_dataloaders()
    # print(summary(model_conv, (3, 224, 224 )))
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    #model = train_classification_model(data, model, criterion, optimizer,
    #                     exp_lr_scheduler, num_epochs=num_epochs)
    train_classification_model_ignite(data, model, criterion, optimizer, num_epochs=num_epochs)

