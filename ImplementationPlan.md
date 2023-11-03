ViT reproduce repo implentation plan:

(start from here) learn how to reproduce the paper https://github.com/UMCai/a-PyTorch-Tutorial-to-Super-Resolution or https://github.com/UMCai/a-PyTorch-Tutorial-to-Machine-Translation

Dataset for classification: 
1. use pytorch build in datasets https://pytorch.org/vision/stable/datasets.html#image-classification
2. dataset should have cache ability https://discuss.pytorch.org/t/best-practice-to-cache-the-entire-dataset-during-first-epoch/19608
3. use default transformation methods, but list all the possibility from transform v2 https://pytorch.org/vision/stable/transforms.html

Model reproduce:
1. pytorch nn.Module (low level)  https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html
2. torchvision,  (mid level) https://pytorch.org/vision/stable/index.html, https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
3. HF transformers  (high level) https://huggingface.co/docs/transformers/main/en/tasks/image_classification
4. Fine-tuning https://pytorch.org/vision/stable/models.html

Training Loop:
1. vanilla pytorch training loop  https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html 
2. pytorch ignite  https://pytorch-ignite.ai/tutorials/
3. different learning rate scheduler + tensorboard  https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html

Evaluation:
1. visualization support
2. inference

proper file structure
1. use argparse & yaml config 
2. hyperparameter search  https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html


The file structure:
1. **tutorials**: includes all non-related tutorials that support this repo
2. **src**: the scripts that are used to support the main.py
    * **model.py**: includes nn.Module pytorch style class, torchvision style class (with pretrain), HF transformers class
    * **trainer.py**: includes pytorch training loop, ignite training loop, with tensorboard and different learning rate scheduler   

3. **demo**: the demo notebook for visualization (all the notebooks are stored here)
4. **main.py**: the entry point 
