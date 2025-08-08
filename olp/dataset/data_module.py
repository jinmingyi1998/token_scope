import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class LengthDataset(Dataset):
    """
    Example dataset class - you need to modify this according to your actual data format
    """

    cls_boundaries = [
        {"cls_id": 0, "start": 0, "end": 128, "range": 128},
        {"cls_id": 1, "start": 129, "end": 256, "range": 128},
        {"cls_id": 2, "start": 257, "end": 512, "range": 256},
        {"cls_id": 3, "start": 513, "end": 1024, "range": 512},
        {"cls_id": 4, "start": 1025, "end": 2048, "range": 1024},
        {"cls_id": 5, "start": 2049, "end": 4096, "range": 2048},
        {"cls_id": 6, "start": 4097, "end": 8192, "range": 4096},
        {"cls_id": 7, "start": 8193, "end": 16384, "range": 8192},
        {"cls_id": 8, "start": 16385, "end": 32768, "range": 16384},
        {"cls_id": 9, "start": 32769, "end": 65536, "range": 32768},
        {"cls_id": 10, "start": 65537, "end": 131072, "range": 65536},
        {"cls_id": 11, "start": 131073, "end": 262144, "range": 131072},
    ]
    num_classes = len(cls_boundaries)

    def __init__(self, data_list: List[Dict[str, Any]], tokenizer, max_length=131072):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def get_cls_index(length):
        for boundary in LengthDataset.cls_boundaries:
            if boundary["start"] <= length <= boundary["end"]:
                return boundary["cls_id"]
        return LengthDataset.num_classes - 1

    @staticmethod
    def get_cls_label_and_soft_target(length):
        cls_label = None
        for boundary in LengthDataset.cls_boundaries:
            if boundary["start"] <= length <= boundary["end"]:
                cls_label = boundary["cls_id"]
                break

        if cls_label is None:
            cls_label = LengthDataset.num_classes - 1

        soft_target = torch.arange(LengthDataset.num_classes, dtype=torch.float32)
        soft_target = torch.where(soft_target < cls_label, 1.0, 0.0)
        soft_target[cls_label] = (
            length - LengthDataset.cls_boundaries[cls_label]["start"]
        ) / LengthDataset.cls_boundaries[cls_label]["range"]

        return cls_label, soft_target

    def __getitem__(self, idx):
        data = self.data_list[idx]
        prompt = data["prompt"]
        length = data["output_length"]
        cls_label, cls_soft_target = self.get_cls_label_and_soft_target(length)

        # Fixed length tokenization - all inputs are padded to 131072 (128k)
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "length": torch.tensor(length, dtype=torch.float32),
            "cls_label": torch.tensor(cls_label, dtype=torch.long),
            "cls_soft_target": torch.tensor(cls_soft_target, dtype=torch.float32),
        }


class LengthDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for length prediction"""

    def __init__(
        self,
        file_list: List[Path],
        tokenizer_name: str = "Qwen/Qwen3-0.6B",
        max_length: int = 10240,
        batch_size: int = 3,
        num_workers: int = 4,
        train_split: float = 0.85,
    ):
        super().__init__()
        self.file_list = file_list
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split

        self.dataset_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "persistent_workers": True if self.num_workers > 0 else False,
        }

        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None

        self.class_counts = [0] * LengthDataset.num_classes
        self.class_weights = [1] * LengthDataset.num_classes

    def setup(self, stage: str = None):
        """Setup datasets"""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, trust_remote_code=True
            )

        if stage == "fit" or stage is None:
            # Load data
            data_list = self.load_data()
            print(f"Loaded {len(data_list)} samples")

            # Split data
            train_data, val_data = train_test_split(
                data_list, test_size=1 - self.train_split, random_state=42
            )

            self.class_weights = self.calculate_class_weights(
                [item["output_length"] for item in train_data]
            )
            print("Class distribution and weights:")
            for i, (count, weight) in enumerate(
                zip(self.class_counts, self.class_weights)
            ):
                print(f"  Class {i}: {count} samples, weight: {weight:.4f}")

            self.train_dataset = LengthDataset(
                train_data, self.tokenizer, self.max_length
            )
            self.val_dataset = LengthDataset(val_data, self.tokenizer, self.max_length)

            print(f"Train samples: {len(self.train_dataset)}")
            print(f"Validation samples: {len(self.val_dataset)}")
            print(f"Using max_length: {self.max_length}")

    def calculate_class_weights(self, lengths: List[int]):
        """Calculate weights based on class counts"""

        for l in lengths:
            self.class_counts[LengthDataset.get_cls_index(l)] += 1
        total_samples = len(lengths)
        self.class_weights = [
            total_samples / count for count in self.class_counts if count > 0
        ]
        return self.class_weights

    def load_data(self):
        """Load data from files"""

        data_list: List[Dict[str, Any]] = []
        for filepath in self.file_list:
            if filepath.exists():
                with filepath.open("r") as f:
                    json_data = json.load(f)
                    for item in json_data:
                        if item.get("input_length", 1) > 10240:
                            continue
                        data_list.append(
                            {
                                "prompt": item["prompt"],
                                "output_length": item["output_length"],
                                "input_length": item["input_length"],
                            }
                        )
            else:
                raise FileNotFoundError(f"File not found: {filepath.absolute()}")
        return data_list

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.dataset_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self.dataset_kwargs)
