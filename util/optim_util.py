import math
import numpy as np
import torch.optim as optim

from constants   import *


def set_optimizer(opt, model):
    if opt.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              opt.lr,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay,
                              dampening=opt.sgd_dampening)
    elif opt.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               opt.lr,
                               betas=(0.9, 0.999),
                               weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(),
                               opt.lr,
                               betas=(0.9, 0.999),
                               weight_decay=opt.weight_decay)
    else:
        raise ValueError(f'Unsupported optimizer: {opt.optimizer}')

    return optimizer



def set_schedule(obj, optim):
        # Set scheduler
        if obj.hparams.scheduler == "warmup":

            def lambda_lr(epoch):
                if epoch <= 3:
                    return 0.001 + epoch * 0.003
                if epoch >= 22:
                    return 0.01 * (1 - epoch / 200.0) ** 0.9
                return 0.01

            scheduler = optim.lr_scheduler.LambdaLR(optim, lambda_lr)
            return optim, scheduler
        elif obj.hparams.scheduler == "cos":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10)
            return optim, scheduler
        elif obj.hparams.scheduler == "plateau":
            data_len = len(obj.train_dataloader())
            val_timing = data_len * obj.hparams.val_check_interval
            scheduler = {
                # patience = every 2 val epochs
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                    optim, factor=0.9, patience=val_timing * 5, min_lr=1e-5
                ),
                "reduce_on_plateau": True,
                "monitor": "val_checkpoint_on",
                "interval": "step",
                "frequency": 1,  # idk if this is working
                "name": "learning_rate",
            }
            return [optim], [scheduler]
        else:
            return optim

