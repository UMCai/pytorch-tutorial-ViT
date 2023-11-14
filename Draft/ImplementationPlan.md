ViT reproduce repo implentation plan:

(start from here) [x] learn how to reproduce the paper https://github.com/UMCai/a-PyTorch-Tutorial-to-Super-Resolution or https://github.com/UMCai/a-PyTorch-Tutorial-to-Machine-Translation

Dataset for classification: 
1. [x] use pytorch build in datasets https://pytorch.org/vision/stable/datasets.html#image-classification
2. dataset should have cache ability https://discuss.pytorch.org/t/best-practice-to-cache-the-entire-dataset-during-first-epoch/19608
3. [x]use default transformation methods, but list all the possibility from transform v2 https://pytorch.org/vision/stable/transforms.html

Model reproduce:
1. [x]pytorch nn.Module (low level)  https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html
2. [x]torchvision,  (mid level) https://pytorch.org/vision/stable/index.html, https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
3. HF transformers  (high level) https://huggingface.co/docs/transformers/main/en/tasks/image_classification
4. [x]Fine-tuning https://pytorch.org/vision/stable/models.html

Training Loop:
1. [x]vanilla pytorch training loop  https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html 
2. [x]pytorch ignite  https://pytorch-ignite.ai/tutorials/
3. [x]different learning rate scheduler + tensorboard  https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html

Evaluation:
1. [x]visualization support
2. [x]inference

proper file structure
1. use argparse & yaml config 
2. hyperparameter search  https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html


The file structure:
1. **tutorials**: includes all non-related tutorials that support this repo
2. **python scripts**: the scripts that are used to support the main.py
    * **Model.py**: includes nn.Module pytorch style class, torchvision style class (with pretrain), HF transformers class
    * **vit_model_reproduce.py**: includes pytorch code for reproduce ViT
    * **test_vit.ipynb**: a testing notebook for checking if the function is correct in **vit_model_reproduce.py**
    * **Trainer.py**: includes pytorch training loop, ignite training loop, with tensorboard and different learning rate scheduler 
    * **Dataset.py**: includes loading the dataset and getting dataloader function
    * **Utils.py**: for visualization  

3. **Config.py**: all the hyperparams are stored here, before running **main.py**, please check if all the config is correct.
4. **main.py**: the entry point 
