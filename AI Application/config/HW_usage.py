import os
import torch

def set_hardware_usage():
    if torch.cuda.is_available():
        os.environ["USE_GPU"] = "1"
        print("GPU is available and will be used.")
    else:
        os.environ["USE_GPU"] = "0"
        print("GPU is not available, using CPU.")