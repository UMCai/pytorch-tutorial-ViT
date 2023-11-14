from torchvision import models
import torch.nn as nn
from vit_model_reproduce import ViT




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



def ViT_reproduce_t_16(img_size = 224, patch_size = 16, num_hiddens = 512, 
                       mlp_num_hiddens = 2048, num_heads = 8, 
                       num_blks = 6, emb_dropout = 0.1,
                       blk_dropout = 0.1, num_ft = 2):
    model = ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
            num_blks, emb_dropout, blk_dropout, num_classes = num_ft)
    return model