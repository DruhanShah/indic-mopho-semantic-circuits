import os
import numpy as np
import torch


def init_environ(scratch: str):

    os.environ["HF_HOME"] = scratch + "/cache/hf"
    os.environ["UV_CACHE_DIR"] = scratch + "/cache/uv"


def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
