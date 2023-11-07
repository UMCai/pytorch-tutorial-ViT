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

By changing the order of tensor operation slightly, we can achieve the same thing in a different way. 
* Firstly, remember that $W_{linear} \in  \mathbb{R}^{D_{patch} \times D}$, this can be reformed as a tensor $ W_{linear} = W_{reform} \in \mathbb{R}^{(P \times P \times 3) \times D} = \mathbb{R}^{3 \times D \times (P \times P)}$. Does this look familiar to you? **This is a 2D convolution layer with input channel = 3, output channel = D, and kernel size = (P, P)** 
* So the input image has shape $X \in \mathbb{R}^{3 \times H \times W}$, but reshaping it into $X_{conv} \in \mathbb{R}^{(\frac{H}{P} \times \frac{W}{P}) \times (P \times P \times 3)}$. By tensor multiplication between $X_{conv}$ and $W_{reform}$, the result stays the same.
$$X_{conv} \times W_{reform} \in  \mathbb{R}^{n_p \times (P \times P \times 3)} \times \mathbb{R}^{(P \times P \times 3) \times D} = \mathbb{R}^{n_p \times D}$$

### 2.2 Position embedding
There are a lot of different methods to embed the position, but here, we simple use the most straightforward one, random value that can be learnt during the training. For each token($n_p$ patches, and one class token), we create random series to add into the patch embedding. Be careful, here we use add, not concat.

### 2.3 Transformer Encoder
In this section, we will start with answering 
1. what is scaled dot-product attention, 
2. what is multi-head attention,
3. what does MLP look like,
4. why using layernorm.

#### 2.3.1 What is scaled dot-product attention
Check *class ScaledDotProductAttention* in *vit_model_reproduce.py*. 

In general, you need to have three matrix: key [n,d], query [m,d], and value [m,v]. 

* key and query con be treated as a key, lock pair. So assuming if you have n different keys with m different locks, we want to know which key is paired with which lock (query). How we can decide if key and lock is a match? We use dot-product. For any one key and one lock, there is a [d,] vector, by dot-product, we will output just a single value, a value reach its max when key and query are in the same direction, and reach its min when the key and query are in opposite direction. For example, $k = [0.5, 0.5], q1 = [1, 1], q2 = [1,-1], q3 = [-1,-1]$, the dot-product between k and q1 is the biggest (because they are towards the same direction 45 degree rotated from X-axis); the dot-product between k and q2, the value is 0, because they are perpendicular, but that's not the worst; the dot-product between k and q3 is negative, which is the smallest, (because they are in the opposite direction).

#### 2.3.2 what is multi-head attention
check *class MultiHeadAttention* in *vit_model_reproduce.py*

Always remember that multi-head attention did not change the shape of tensor! And the number of heads needs to be divided by feature number! 

In general, multi-head is a way to let the attention mechanism works with multiply channels, each channel can pay attention to different features or patterns. Just like convolution can learn different filters to detect edges, multi-head attention can do the same, but better (because there is no locality as pre-knowledge). From the figure below, there are three key points:
1. attention has no locality, this means even at the shallow layer of neural network, two distant region are pay attnetion with each other.
2. convolution has inductive bias and locality. This is why convolution can only learn global features at the deep layers of neural network.
3. 

<p align="center">
<img src="./img/multihead_attention.png">
</p>
