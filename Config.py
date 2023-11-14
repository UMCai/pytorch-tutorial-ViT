import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
HYMENOPTERA_MODEL_PATH = r"C:\Users\Shizh\OneDrive - Maastricht University\Data\hymenoptera_data\Model_pt"
DOGTINY_MODEL_PATH = r"C:\Users\Shizh\OneDrive - Maastricht University\Data\kaggle_dog_tiny\Model_pt"
HYMENOPTERA_DATA_PATH = r"C:\Users\Shizh\OneDrive - Maastricht University\Data\hymenoptera_data"
DOGTINY_DATA_PATH =     r"C:\Users\Shizh\OneDrive - Maastricht University\Data\kaggle_dog_tiny"

HYMENOPTERA_FIGURE_PATH = r"C:\Users\Shizh\OneDrive - Maastricht University\Data\hymenoptera_data\figures"
DOGTINY_FIGURE_PATH = r"C:\Users\Shizh\OneDrive - Maastricht University\Data\kaggle_dog_tiny\figures"

IMG_SIZE = 224  # this is the input image size for ViT



