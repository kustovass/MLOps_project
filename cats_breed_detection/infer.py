import pytorch_lightning as pl

from cat_breed_detection.dataclass import CatDataModule
from cat_breed_detection.model import CatModel


def infer(cfg):
    model = CatModel.load_from_checkpoint(checkpoint_path=cfg.model.save_model_name)

    dm = CatDataModule(cfg)
    dm.setup(stage="predict")
    test_loader = dm.test_dataloader()

    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
    )

    trainer.predict(model, test_loader)
