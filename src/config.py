import torch

# Training Hyperparameters
INPUT_SIZE = 120
NUM_CLASSES = 2
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
NUM_EPOCHS = 100
PATIENCE = 20

# Dataset
DATA_DIR = "./STAN_patches_lbls"
NUM_WORKERS = 4

# Compute
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICES = [0]
