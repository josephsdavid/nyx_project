from argparse import ArgumentParser
from args import LightningBaseArgParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateLogger

from lightning.example_lightning import ExampleModel


seed_everything(6)

def main(args):
    logger = pl_loggers.WandbLogger(experiment="example", save_dir=None)
    early_stop = EarlyStopping(
        monitor="val_loss"
    )
    checkpoint_callback = ModelCheckpoint(dirpath="ckpts/", monitor="val_loss")
    model = ExampleModel(args)
    lr_logger = LearningRateLogger()
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks =  [early_stop, lr_logger],
        checkpoint_callback=checkpoint_callback
    )
    trainer.fit(model)


if __name__ == "__main__":
    parser = LightningBaseArgParser().get_parser()
    parser = ExampleModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
