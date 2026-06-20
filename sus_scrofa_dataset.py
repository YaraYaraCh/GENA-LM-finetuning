#!/usr/bin/env python3
"""
PyTorch Dataset для чтения HDF5 датасета промоторов свинки.

Каждый сэмпл возвращает:
    input_ids      : torch.LongTensor shape (seq_len,)
    attention_mask : torch.LongTensor shape (seq_len,)
    labels         : int  1 = промотор (_pos),  0 = антипромотор (_neg)
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SusScrofaPromoterDataset(Dataset):
    def __init__(self, h5_path, max_seq_len=2000):
        """
        Args:
            h5_path    : путь к HDF5 файлу (train / valid / test)
            max_seq_len: максимальная длина последовательности (default: 2000)
        """
        self.h5_path     = str(h5_path)
        self.max_seq_len = max_seq_len
        with h5py.File(self.h5_path, "r") as f:
            self.keys = sorted(
                f.keys(),
                key=lambda x: (int(x.split("_")[0]), x.split("_")[1])
            )
        self.length = len(self.keys)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        key = self.keys[idx]
        with h5py.File(self.h5_path, "r") as f:
            grp = f[key]
            input_ids      = grp["input_ids"][0].astype(np.int64)
            attention_mask = grp["attention_mask"][0].astype(np.int64)
            sample_type = grp.attrs.get("type", "")
            label = 1 if sample_type == "promoter" else 0

        input_ids      = input_ids[:self.max_seq_len]
        attention_mask = attention_mask[:self.max_seq_len]

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(label,          dtype=torch.long),
        }