from argparse import ArgumentParser
from args import LightningBaseArgParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from lightning import AutoencoderModel


seed_everything(6)

def main(args):
    logger = pl_loggers.WandbLogger(experiment=None, save_dir=None)
    checkpoint_callback = ModelCheckpoint(dirpath=f"ckpts/{args.reduction}_reduction/", monitor="val_loss")
    model = AutoencoderModel(args)
    lr_logger = LearningRateMonitor()
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks =  [lr_logger],
        checkpoint_callback=checkpoint_callback
    )
    trainer.fit(model)


if __name__ == "__main__":
    parser = LightningBaseArgParser().get_parser()
    parser = AutoencoderModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)

