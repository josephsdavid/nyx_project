import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import util
from argparse import ArgumentParser
from dataset import get_dataloader
from models import AutoEncoder


class ExampleModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.hparams["tpu_cores"] = 0
        self.loss = self.get_loss_fn()
        # you can get fancier here of course, we will likely have a separate
        # class for the model
        self.model = AutoEncoder(
            hparams["latent_dim"], hparams["outer_dim"], hparams["shrink_factor"]
        )

    def forward(self, inputs):
        output = self.model(inputs)
        return dict(latent=output[0], predicted=output[1])

    def training_step(self, batch, batch_idx):
        x = batch
        sign = torch.sign(x)
        _, preds = self.model(x)
        preds = preds * sign
        loss = self.loss(preds, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        sign = torch.sign(x)
        _, preds = self.model(x)
        loss = self.loss(preds * sign, x)
        self.log("val_loss", loss)
        for n in [1, 5, 10, 20]:
            x_mask = x.clone().detach()
            for i in range(x_mask.shape[0]):
                num_revs = x_mask[i, :].bool().sum()
                if n > num_revs:
                    x_mask[i, :] = 0
                else:
                    x_mask[i, :][torch.where(x_mask[i, :] > 0)[-n:]] = 0
            _, preds = self.model(x_mask)
            loss = self.loss(preds * sign, x)
            self.log(f"val_last_{n}_loss", loss)

        def get_loss_fn(self):
            loss = nn.MSELoss()
            return loss

    def configure_optimizers(self):
        if self.hparams["optimizer"] == "Adam":
            optim = torch.optim.Adam(self.model.parameters(), lr=self.hparams["lr"])
        else:
            optim = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hparams["lr"],
                momentum=self.hparams["momentum"],
            )

        # test this
        return util.set_schedule(self, optim)

    def __dataloader(self, split):
        return get_dataloader(split)

    def val_dataloader(self):
        return self.__dataloader("valid", self.hparams)

    def train_dataloader(self):
        return self.__dataloader("train")

    def test_dataloader(self):
        return self.__dataloader("test")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--latent_dim", type=int, default=256)
        parser.add_argument("--outer_dim", type=int, default=2048)
        parser.add_argument("--shrink_factor", type=int, default=2)
