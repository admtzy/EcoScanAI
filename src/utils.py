from pathlib import Path
import random
import numpy as np
import tensorflow as tf
from config import SEED

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def create_folder(path):
    Path(path).mkdir(
        parents=True,
        exist_ok=True
    )


def print_header(title):
    print("=" * 60)
    print(title)
    print("=" * 60)