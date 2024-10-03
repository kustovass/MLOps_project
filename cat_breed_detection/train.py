import os

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from cats_breed_detection.dataclass import CatDataModule
from cats_breed_detection.model import CatModel


# Create train and validation datasets
# TRAIN_DIR = Path("data/train")


def train(cfg: DictConfig):
    pl.seed_everything(0)
    torch.set_float32_matmul_precision("medium")
    dm = CatDataModule(cfg)
    model = CatModel(cfg)

    loggers = [
        pl.loggers.CSVLogger("./.logs/my-csv-logs", name=cfg.artifacts.experiment_name),
        pl.loggers.MLFlowLogger(
            experiment_name=cfg.artifacts.experiment_name,
            tracking_uri="file:./.logs/my-mlflow-logs",
        ),
    ]

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(max_depth=cfg.callbacks.model_summary.max_depth),
    ]

    if cfg.callbacks.swa.use:
        callbacks.append(
            pl.callbacks.StochasticWeightAveraging(swa_lrs=cfg.callbacks.swa.lrs)
        )

    if cfg.artifacts.checkpoint.use:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(
                    cfg.artifacts.checkpoint.dirpath, cfg.artifacts.experiment_name
                ),
                filename=cfg.artifacts.checkpoint.filename,
                monitor=cfg.artifacts.checkpoint.monitor,
                mode=cfg.artifacts.checkpoint.mode,
                save_top_k=cfg.artifacts.checkpoint.save_top_k,
                every_n_epochs=cfg.artifacts.checkpoint.every_n_epochs,
                verbose=True,
            )
        )

    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        accumulate_grad_batches=cfg.train.grad_accum_steps,
        max_epochs=cfg.train.epochs,
        val_check_interval=cfg.train.val_check_interval,
        overfit_batches=cfg.train.overfit_batches,
        num_sanity_val_steps=cfg.train.num_sanity_val_steps,
        deterministic=cfg.train.full_deterministic_mode,
        benchmark=cfg.train.benchmark,
        gradient_clip_val=cfg.train.gradient_clip_val,
        profiler=cfg.train.profiler,
        log_every_n_steps=cfg.train.log_every_n_steps,
        detect_anomaly=cfg.train.detect_anomaly,
        enable_checkpointing=cfg.artifacts.checkpoint.use,
        logger=loggers,
        callbacks=callbacks,
    )

    if cfg.train.batch_size_finder:
        tuner = pl.tuner.Tuner(trainer)
        tuner.scale_batch_size(model, datamodule=dm, mode="power")

    trainer.fit(model, datamodule=dm)

    trainer.save_checkpoint(cfg.model.save_model_name)
