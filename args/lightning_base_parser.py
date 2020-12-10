import os
import sys
import torch
import argparse
sys.path.append(os.getcwd())

from constants import *
from util import str2bool


class LightningBaseArgParser:
    '''Base training argument parser
    '''

    def __init__(self):
        self.parser = argparse.ArgumentParser(description = "Base Arguments")

        # logger args
        self.parser.add_argument("--wandb_project_name", type=str, default="debug")
        self.parser.add_argument("--trial_name", type=str, default='trial_1.0')
        self.parser.add_argument("--save_dir", type=str, default=LOG_DIR)
        self.parser.add_argument("--experiment_name", type=str, default='debug')

        # training
        self.parser.add_argument('--lr', type=float, default=0.001)
        self.parser.add_argument("--batch_size", type=int, default=128)
        self.parser.add_argument("--num_workers", type=int, default=8)
        self.parser.add_argument('--optimizer', type=str, default='AdamW')
        self.parser.add_argument("--lr_decay", type=float, default=0.1)
        self.parser.add_argument("--weight_decay", type=float, default=0.0)
        self.parser.add_argument("--momentum", type=float, default=0.9)
        self.parser.add_argument("--sgd_dampening", type=float, default=0.9)

        # dataset and augmentations
        self.parser.add_argument("--img_type", type=str, default="Frontal", choices=["All", "Frontal", "Lateral"])
        self.parser.add_argument("--uncertain", type=str, default="ignore", choices=["ignore", "zero", "one"])
        self.parser.add_argument("--resize_shape", type=int, default=256)
        self.parser.add_argument("--crop_shape", type=int, default=224)
        self.parser.add_argument("--rotation_range", type=int, default=20)
        self.parser.add_argument("--gaussian_noise_mean", type=float, default=None)
        self.parser.add_argument("--gaussian_noise_std", type=float, default=None)
        self.parser.add_argument("--gaussian_blur_radius", type=float, default=None)

    def get_parser(self):
        return self.parser
