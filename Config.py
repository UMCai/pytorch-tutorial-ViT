import torch
import Model 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
HYMENOPTERA_MODEL_PATH = r"C:\Users\Shizh\OneDrive - Maastricht University\Data\hymenoptera_data\Model_pt"
DOGTINY_MODEL_PATH = r"C:\Users\Shizh\OneDrive - Maastricht University\Data\kaggle_dog_tiny\Model_pt"
HYMENOPTERA_DATA_PATH = r"C:\Users\Shizh\OneDrive - Maastricht University\Data\hymenoptera_data"
DOGTINY_DATA_PATH =     r"C:\Users\Shizh\OneDrive - Maastricht University\Data\kaggle_dog_tiny"

HYMENOPTERA_FIGURE_PATH = r"C:\Users\Shizh\OneDrive - Maastricht University\Data\hymenoptera_data\figures"
DOGTINY_FIGURE_PATH = r"C:\Users\Shizh\OneDrive - Maastricht University\Data\kaggle_dog_tiny\figures"

IMG_SIZE = 224  # this is the input image size for ViT

MODE = 'training' 
# DATA_NAME = 'hymenoptera' 
DATA_NAME = 'dogtiny'
NUM_CLASSES = 120 if DATA_NAME == 'dogtiny' else 2
MODEL_PATH = DOGTINY_MODEL_PATH if DATA_NAME == 'dogtiny' else HYMENOPTERA_DATA_PATH
FIGURE_PATH = DOGTINY_FIGURE_PATH if DATA_NAME == 'dogtiny' else HYMENOPTERA_FIGURE_PATH
# MODE = 'inference'
BATCH_SIZE = 4
NUM_WORKERS = 2
NUM_EPOCHS = 5
LR = 0.001

#MODEL = Model.ViT_reproduce_t_16(IMG_SIZE)  
MODEL = Model.ViT_b_16(num_ft=NUM_CLASSES)