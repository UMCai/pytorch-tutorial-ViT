from torchvision import models
import torch.nn as nn

#############
# ResNet (torchvision)
# we start from a simple convolutional net to fit the pipeline
def ResNet18(num_ft = 2):
    model_conv = models.resnet18(weights='IMAGENET1K_V1')
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, num_ft)
    return model_conv
#############


#############
# ViT (torchvision)
# this ViT is directly used from the torchvision, for fine-tuning
def ViT_b_16(num_ft = 2):
    model_vit = models.vit_b_16(weights = models.ViT_B_16_Weights.IMAGENET1K_V1)
    for param in model_vit.parameters():
        param.requires_grad = False
    num_ftrs = model_vit.heads.head.in_features
    model_vit.heads.head =  nn.Linear(num_ftrs, num_ft)
    return model_vit
#############