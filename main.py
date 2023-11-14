import warnings
warnings.filterwarnings("ignore")
import yaml
import os
import Config
from Dataset import dogtiny_dataloaders, hymenoptera_dataloaders
from Trainer import train_classification_model, train_classification_model_ignite
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import Model
cudnn.benchmark = True

def load_config(config_name):
    with open(os.path.join('configs', config_name)) as file:
        config = yaml.safe_load(file)
    return config


def get_argparse():
    import argparse
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="the config file name")
    args = vars(ap.parse_args())
    return args

def main():
    device = Config.DEVICE
    args = get_argparse()
    cfg = load_config(args['config'])

    data_name = cfg['data_name']
    num_ft = cfg['num_classes']
    num_epochs = cfg['num_epochs']
    lr = cfg['lr']
    mode = cfg['mode']

    if data_name=='dogtiny':
        model_checkpoint_path = Config.DOGTINY_MODEL_PATH 
        data = dogtiny_dataloaders(cfg['batch_size'],cfg['num_workers'])
        figure_path = Config.DOGTINY_FIGURE_PATH
    if data_name=='hymenoptera':
        model_checkpoint_path = Config.HYMENOPTERA_MODEL_PATH 
        data = dogtiny_dataloaders(cfg['batch_size'],cfg['num_workers'])
        figure_path = Config.HYMENOPTERA_FIGURE_PATH
    
    if cfg['model_name'] == 'ViT_b_16':
        model = Model.ViT_b_16(num_ft=num_ft)
    if cfg['model_name'] == 'ViT_reproduce_t_16':
        model = Model.ViT_reproduce_t_16(num_ft=num_ft)
    if cfg['model_name'] == 'ResNet18':
        model = Model.ResNet18(num_ft=num_ft)


    if mode == 'training':
        print('mode is:', mode)
        criterion = nn.CrossEntropyLoss()
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        #model = train_classification_model(data, model, criterion, optimizer,
        #                     exp_lr_scheduler,model_checkpoint_path, num_epochs=num_epochs)
        train_classification_model_ignite(data, model, criterion, optimizer, model_checkpoint_path, scheduler=exp_lr_scheduler, num_epochs=num_epochs)
    
    if mode == 'inference':
        print('mode is:', mode)
        from Utils import visualize_model
        import os
        model.load_state_dict(torch.load(os.path.join(model_checkpoint_path, 'best_model_1_accuracy=0.9869.pt')))
        visualize_model(data, model, figure_path, num_images = cfg['batch_size'])



if __name__ == '__main__':
    main()


