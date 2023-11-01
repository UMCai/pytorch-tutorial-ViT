import torch
import Model 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"C:\Users\Shizh\OneDrive - Maastricht University\Data\hymenoptera_data\Model_pt"
DATA_PATH = r"C:\Users\Shizh\OneDrive - Maastricht University\Data\hymenoptera_data"
FIGURE_PATH = r"C:\Users\Shizh\OneDrive - Maastricht University\Data\hymenoptera_data\figures"

BATCH_SIZE = 4
NUM_EPOCHS = 10
LR = 0.001

MODEL = Model.ViT_b_16()