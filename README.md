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
(B = batch size, 3 = # of RGB channels, H, W = Height, width, nP = # of patches, D(d) = # of hiddens)

0. input images with shape **[B, 3, H, W]**
1. devide each image into small patches *class PatchEmbedding*, embedded patches with shape **[B, P, d]**
2. concat a *cls_token* to the embedded patches, the shape will be **[B, P+1, d]**
3. add (not concat) *position_embedding* to it **[B, P+1, d]**
4. feed the embedded patches to the transformer encoder (several blocks of *class ViTBlock*), the output shape after each transformer layers stays the same **[B, P+1, d]**
5. take the features (X -> [B, P+1, d]) related to class token **X[:, 0, :]**, get output as shape **[B, 1, d] = [B, d]**
6. feed into *MLP_head* **[B, # of class]**

### 2.1 PatchEmbedding
Check *class PatchEmbedding* in *vit_model_reproduce.py*, and test the complete code in *test_vit.ipynb*. (P = patch size)

In this part, the original paper, they said:
>The Transformer uses constant latent vector size D through all of its layers, so we
flatten the patches and map to D dimensions with a trainable linear projection.

There are two steps here:
1. (create all the patches), and then flatten all the patches (treat each patch as a token in NLP)
2. linear project to $D$ dimensions

But mathmatically, these two steps can be done by a simple 2D convolution operation. By using 
~~~
nn.Conv2d(in_channels = 3, out_channels = D, kernel_size = (P, P), stride = P, padding = 0)
~~~
But why they are equivalent? Let's prove it!
* Firstly we will see how the patch works. let's assume the input image is $X \in \mathbb{R}^{3 \times H \times W}$. By creating $P \times P$ patches, $X$ is reshaped as $X \in \mathbb{R}^{3 \times (\frac{H}{P} \times P) \times (\frac{W}{P} \times P)}$, and further into $X \in \mathbb{R}^{(\frac{H}{P} \times \frac{W}{P}) \times (P \times P \times 3)}$. Since the number of patches is $n_{p} = \frac{H}{P} \times \frac{W}{P}$, the whole thing can be simplified as $X \in \mathbb{R}^{n_{p} \times (P \times P \times 3)}$. After flatten each patches, assume that $D_{patch} = P \times P \times 3$, we have a tensor $X_{flatten} \in \mathbb{R}^{n_{p} \times D_{patch}}$
* Next, let's see how linear project works. By using a linear layer that maps $D_{patch}$ dimension to $D$ dimension, the weight should be $W_{linear} \in \mathbb{R}^{D_{patch} \times D}$. So by using matrix multiplication $X_{flatten} \times W_{linear} \in \mathbb{R}^{n_{p} \times D_{patch}} \times \mathbb{R}^{D_{patch} \times D} = \mathbb{R}^{n_{p} \times D}$, and here we get the embedded $n_{p}$ patches with feature $D$. 

By changing slightly, 


