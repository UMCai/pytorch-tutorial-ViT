# pytorch-tutorial-ViT

This repo aims to reproduce the code from [paper](https://arxiv.org/abs/2010.11929). During this tutorial you will learn:
1. the details about **transformer** ([paper](https://arxiv.org/abs/1706.03762))
2. the details about **vision transformer**
3. how to fine-tune ViT by using **torchvision** and **HF transformers**   
4. how to use [**ignite**](https://pytorch-ignite.ai/) to train and evaluate the model


## 0. Environment setup:

Use conda to create the virtual env, all the python packages are explicitly stored within.

Open conda prompt and type the following command:
```
conda create --name myenv --file requirements_torch_monai_env.txt
```

## 1. Prepare dataset
In this tutorial, we choose to use a very simple classification dataset -- hymenoptera dataset, the download link is [here](https://download.pytorch.org/tutorial/hymenoptera_data.zip).

In this simple classiciation task, there are two classes: ants and bees. If you are not familiar with pytorch dataset and dataloader, please check [official pytorch info](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), or *tutorials/basic_pytorch_tutorials/check_different_datasets.ipynb*

Dataset is used to transform the data path into data (tensor). And dataloader is to batchize the dataset samples into a pre-determined batch size and feed to the neural network. Please check *Dataset.py*

**IMPORTANT:** if you want to use a different dataset, be sure to prepare the data with three different features:
1. the dataloaders -- a dict with 'train' and 'val' as keys
2. class_names -- a list includes all the class names with orders
3. dataset_size -- a dict with 'train' and 'val' as keys

## 2. ViT model reproduction
This is the main topic for this tutorial, please check *vit_model_reproduce.py*.

Firstly, let's have a look at the model structure.

<p align="center">
<img src="./img/ViT_model_structure.png">
</p>

A very clear model structure can be seen here, let's focus on these key elements:
1. **Patch**, divide one image into small patches
2. **Position embedding**, a common thing for transformer based model, to learn the location of each patch
3. **Transformer encoder**, the details are shown in the right side, two key things:
    * **Multi-Head Attention**
    * **MLP**, a combination of dropout, linear layer and activation function -- GeLU
    * **Norm**, always LayerNorm
4. **MLP head**, a classification head

### 2.0 The data flow of vision transformer  
(B = batch size, 3 = # of RGB channels, H, W = Height, width, P = # of patches, d = # of hiddens)

0. input images with shape **[B, 3, H, W]**
1. devide each image into small patches *class PatchEmbedding*, embedded patches with shape **[B, P, d]**
2. concat a *cls_token* to the embedded patches, the shape will be **[B, P+1, d]**
3. add (not concat) *position_embedding* to it **[B, P+1, d]**
4. feed the embedded patches to the transformer encoder (several blocks of *class ViTBlock*), the output shape after each transformer layers stays the same **[B, P+1, d]**
5. take the features (X -> [B, P+1, d]) related to class token **X[:, 0, :]**, get output as shape **[B, 1, d] = [B, d]**
6. feed into *MLP_head* **[B, # of class]**

### 2.1 PatchEmbedding
Check *class PatchEmbedding* in *vit_model_reproduce.py*
