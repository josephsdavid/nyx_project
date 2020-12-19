import os
import sys
import torch
import argparse
sys.path.append(os.getcwd())

from util import str2bool


class LightningBaseArgParser:
    '''Base training argument parser
    '''

    def __init__(self):
        self.parser = argparse.ArgumentParser(description = "Base Arguments")

        # logger args
        self.parser.add_argument("--wandb_project_name", type=str, default="debug")
        self.parser.add_argument("--trial_name", type=str, default='trial_1.0')
        self.parser.add_argument("--save_dir", type=str, default="ckpts")
        self.parser.add_argument("--experiment_name", type=str, default='debug')

        # training
        self.parser.add_argument('--lr', type=float, default=0.001)
        self.parser.add_argument("--batch_size", type=int, default=128)
        self.parser.add_argument("--num_workers", type=int, default=8)
        self.parser.add_argument('--optimizer', type=str, default='AdamW')
        self.parser.add_argument("--lr_decay", type=float, default=0.1)
        self.parser.add_argument("--weight_decay", type=float, default=0.0)
        self.parser.add_argument("--momentum", type=float, default=0.9)

    def get_parser(self):
        return self.parser
