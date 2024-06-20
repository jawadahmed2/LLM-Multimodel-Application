import os
import torch
from loguru import logger

def set_hardware_usage():
    if torch.cuda.is_available():
        os.environ["USE_GPU"] = "1"
        logger.info("GPU is available and will be used.")
    else:
        os.environ["USE_GPU"] = "0"
        logger.info("GPU is not available, using CPU.")