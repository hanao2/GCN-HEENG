import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
from utils import Train

def main():
    for i in range(1, 5):
        my_training = Train()
        my_training.train_ensemble(i)

if __name__ == "__main__":
    main()
