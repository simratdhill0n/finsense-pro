import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class DocumentDataset(Dataset):
    def __init__(self, csv_path, tokenizer: PreTrainedTokenizer, label2id: dict, max_length=256):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            str(row["invoice_text"]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.label2id[row["document_type"]], dtype=torch.long),
        }


def build_label_maps(types):
    label2id = {t: i for i, t in enumerate(sorted(types))}
    id2label = {i: t for t, i in label2id.items()}
    return label2id, id2label


def save_label_maps(label2id, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(label2id, indent=2), encoding="utf-8")
