import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from dvc.api import DVCFileSystem
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms


DATA_MODES = ["train", "val", "test"]
RESCALE_SIZE = 224


class CatDataset(Dataset):
    def __init__(self, files, mode):
        self.files = sorted(files)
        self.mode = mode
        self.len_ = len(self.files)
        self.label_encoder = LabelEncoder()

        if self.mode not in DATA_MODES:
            raise NameError(f"{self.mode} is not correct; correct modes: {DATA_MODES}")

        if self.mode != "test":
            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)

            with open("label_encoder.pkl", "wb") as le_dump_file:
                pickle.dump(self.label_encoder, le_dump_file)

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image

    def __getitem__(self, index):
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        x = self.load_sample(self.files[index])
        x = self._prepare_sample(x)
        x = np.array(x / 255, dtype="float32")

        if self.mode == "train":
            x = train_transform(x)
        else:
            x = val_transform(x)

        if self.mode == "test":
            return x
        else:
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
            return x, y

    def _prepare_sample(self, image):
        image = image.resize((RESCALE_SIZE, RESCALE_SIZE))
        return np.array(image)


class CatDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.val_size = (cfg.data.val_size,)
        self.dataloader_num_workers = cfg.data.dataloader_num_workers
        self.batch_size = cfg.data.batch_size
        self.train_dir = Path(cfg.data.train_dir)
        self.test_dir = Path(cfg.data.test_dir)

    def prepare_data(self):
        if (self.test_dir).exists() and (self.train_dir).exists():
            print("Data already exists")
            return super().prepare_data()

        fs = DVCFileSystem(".", subrepos=True)
        fs.get("data", "data", recursive=True)
        print("Data downloaded")
        return super().prepare_data()

        # self.TRAIN_DIR, self.TEST_DIR = load_data()

    def setup(self, stage: Optional[str] = None):
        print("Setup")

        if stage == "fit":
            train_val_files = sorted(list(self.train_dir.rglob("*.jpeg")))

            train_val_labels = [path.parent.name for path in train_val_files]

            if len(train_val_files) == 0:
                raise RuntimeError(
                    "Не найдено файлов для обработки. Проверьте путь к данным."
                )

            train_files, val_files = train_test_split(
                train_val_files, test_size=0.25, stratify=train_val_labels
            )
            self.train_dataset = CatDataset(train_files, mode="train")
            self.val_dataset = CatDataset(val_files, mode="val")

        elif stage == "test" or stage == "predict":
            test_files = sorted(list(self.test_dir.rglob("*.jpeg")))

            self.test_dataset = CatDataset(test_files, mode="test")

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataloader_num_workers,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_workers,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        print("Test_loader")
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_workers,
        )
