import warnings
warnings.filterwarnings("ignore")

import Config
from Dataset import hymenoptera_dataloaders
from Trainer import train_classification_model, train_classification_model_ignite
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
#from torchsummary import summary
cudnn.benchmark = True



def main():
    device = Config.DEVICE
    model = Config.MODEL
    num_epochs = Config.NUM_EPOCHS
    lr = Config.LR
    mode = Config.MODE

    data = hymenoptera_dataloaders()
    # print(summary(model, (3, 224, 224 )))
    
    if mode == 'training':
        print('mode is:', mode)
        criterion = nn.CrossEntropyLoss()
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        #model = train_classification_model(data, model, criterion, optimizer,
        #                     exp_lr_scheduler, num_epochs=num_epochs)
        train_classification_model_ignite(data, model, criterion, optimizer, scheduler=exp_lr_scheduler, num_epochs=num_epochs)
    
    if mode == 'inference':
        print('mode is:', mode)
        from Utils import visualize_model
        import os
        model.load_state_dict(torch.load(os.path.join(Config.MODEL_PATH , 'best_model_1_accuracy=0.9869.pt')))
        visualize_model(data, model, num_images = Config.BATCH_SIZE)



if __name__ == '__main__':
    
    main()


