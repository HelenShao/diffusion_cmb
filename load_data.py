import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages  # Import PdfPages
import seaborn as sns
import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F

import math
import seaborn as sns
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings("ignore")
import sys, os

# Add PartIII-project to the Python path
sys.path.append(os.path.abspath("/scratch/gpfs/hshao/PartIII-Project"))
from cmb_tools_main import *

# Add the path of 'folder1' to the system path
files_dir = "/scratch/gpfs/hshao/ILC_ML/reproducing/unet_vanilla_4_freq"
sys.path.append(os.path.abspath(files_dir))

import loss_funcs
import data
import common_vars
from common_vars import *

# Function to set the seed for reproducibility
def set_seed(seed):
    # Set seed for PyTorch, NumPy, and Python's random module
    print(f"Set seed to {seed} for torch, numpy, and python random")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # For multi-GPU, make sure all GPUs are initialized with the same seed
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior for data loading across multiple workers
    def worker_init_fn(worker_id):
        seed_all = seed + worker_id
        np.random.seed(seed_all)
        random.seed(seed_all)
    
    return worker_init_fn

# Set the seed
seed = 4
worker_init_fn = set_seed(seed)

# Use GPUs if avaiable
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

normalize=False
transform=None
remove_mean=False

config = {
        "batch_size": 32,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "max_epochs": 700,
        "seed": 4,
        "in_channels": 1,
        "out_channels": 1,
        "feature_dims": [32, 64, 128, 256, 512, 1024],
        "directory": f'/scratch/gpfs/hshao/ILC_ML/1_freq_scale',
        "transform": None,
    }

num_epochs   = config["max_epochs"]
batch_size   = config["batch_size"]
seed         = config["seed"]
in_channels  = config["in_channels"]
out_channels = config["out_channels"]
feature_dims = config["feature_dims"]
transform    = config["transform"]
directory    = config["directory"]
num_workers  = int(os.cpu_count()/2)
ell_cutoff   = 200
dataset_path = f"{directory}/X_Y_dataset_nside_{nside}_1freq_220.0_fg_ell{ell_cutoff}.pkl"

# Single frequency only
freq_names = ['220.0 GHz']

try:
    with open(dataset_path, 'rb') as file:
        X_Y_dataset = pickle.load(file)
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    
data_splits = data.organize_data_4(freq_names, X_Y_dataset, train_ratio=0.8)

# Create datasets
train_dataset = data.CMBPatchesDataset(image_dict=data_splits, freq_names=freq_names, dataset_type='train', transform=transform, normalize=normalize, remove_mean=remove_mean)

# Get the mean and std from the training set
if normalize:
    mean_input, std_input = train_dataset.mean_input, train_dataset.std_input
    mean_target, std_target = train_dataset.mean_target, train_dataset.std_target
else:
    mean_input, std_input = None, None
    mean_target, std_target = None, None


# Validation 
valid_dataset = data.CMBPatchesDataset(image_dict=data_splits, freq_names=freq_names, 
                                       dataset_type='valid',
                                       mean_input=mean_input, 
                                       std_input=std_input, 
                                       mean_target=mean_target, 
                                       std_target=std_target,
                                        normalize=normalize, remove_mean=remove_mean)

# Test
test_dataset = data.CMBPatchesDataset(image_dict=data_splits, freq_names=freq_names, 
                                 dataset_type='test', 
                                 mean_input=mean_input, 
                                 std_input=std_input, 
                                 mean_target=mean_target, 
                                 std_target=std_target, 
                                      normalize=normalize, remove_mean=remove_mean)

datasets = {'train': {}, 'valid': {}, 'test': {}, }
datasets['train'] = train_dataset
datasets['valid'] = valid_dataset
datasets['test'] = test_dataset

train_loader, valid_loader, test_loader = data.create_loaders(datasets, seed, batch_size, num_workers, worker_init_fn=worker_init_fn)

