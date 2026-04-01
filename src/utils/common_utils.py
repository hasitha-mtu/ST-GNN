import torch
import numpy as np
import random
import os

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if using multi-GPU

    # Force deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Required for PyTorch 1.7+ to handle non-deterministic GNN operations
    torch.use_deterministic_algorithms(True)

    # For PyTorch Geometric specifically
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

