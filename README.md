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
conda create --name myenv --file requirements.txt
```