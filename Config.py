import torch
import Model 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"C:\Users\Shizh\OneDrive - Maastricht University\Data\hymenoptera_data\Model_pt"
HYMENOPTERA_DATA_PATH = r"C:\Users\Shizh\OneDrive - Maastricht University\Data\hymenoptera_data"
FIGURE_PATH = r"C:\Users\Shizh\OneDrive - Maastricht University\Data\hymenoptera_data\figures"
IMG_SIZE = 224

MODE = 'training' 
# MODE = 'inference'
BATCH_SIZE = 4
NUM_WORKERS = 2
NUM_EPOCHS = 10
LR = 0.001

MODEL = Model.ViT_reproduce_t_16(IMG_SIZE)  
#MODEL = Model.ViT_b_16()