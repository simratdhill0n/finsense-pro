"""
Task 2 — DistilBERT document classifier (4 classes).

Before training:
  1. Put DistilBERT files in models/pretrained_distilbert/  (see README there)
  2. python scripts/check_distilbert.py
  3. python scripts/setup_data.py && python scripts/create_splits.py
  4. python train_document_classifier.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.document_dataset import DocumentDataset, build_label_maps, save_label_maps

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PRETRAINED = ROOT / "models" / "pretrained_distilbert"
MODEL_DIR = ROOT / "models" / "document_classifier"
RESULTS = ROOT / "results" / "task2"

SEED = 42
MAX_LEN = 256
BATCH = 32
EPOCHS = 3
LR = 2e-5


def check_weights():
    weights = PRETRAINED / "pytorch_model.bin"
    if not weights.exists():
        raise FileNotFoundError(
            f"Missing {weights}\n"
            "Download DistilBERT manually — see models/pretrained_distilbert/README.md"
        )
    size_mb = weights.stat().st_size / 1_000_000
    if size_mb < 250:
        raise FileNotFoundError(
            f"{weights} is only {size_mb:.0f} MB (need ~268 MB). Re-download the file."
        )


def set_seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


@torch.no_grad()
def evaluate(model, loader, device, id2label):
    model.eval()
    yt, yp = [], []
    for batch in loader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).logits
        yp.extend(logits.argmax(-1).cpu().tolist())
        yt.extend(batch["labels"].cpu().tolist())
    names = [id2label[i] for i in range(len(id2label))]
    return {
        "accuracy": accuracy_score(yt, yp),
        "f1_macro": f1_score(yt, yp, average="macro"),
        "f1_weighted": f1_score(yt, yp, average="weighted"),
        "report": classification_report(yt, yp, target_names=names, output_dict=True),
        "y_true": yt,
        "y_pred": yp,
    }


def plot_cm(yt, yp, labels, path):
    cm = confusion_matrix(yt, yp)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("DistilBERT — Test set")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    check_weights()
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    for name in ("train.csv", "val.csv", "test.csv"):
        if not (DATA / name).exists():
            raise FileNotFoundError(f"Missing data/{name}. Run scripts/create_splits.py")

    train_df = pd.read_csv(DATA / "train.csv")
    label2id, id2label = build_label_maps(train_df["document_type"].unique())
    print("Classes:", label2id)

    print("Loading DistilBERT from", PRETRAINED)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)
    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED,
        num_labels=len(label2id),
        id2label={str(k): v for k, v in id2label.items()},
        label2id=label2id,
    ).to(device)

    train_loader = DataLoader(
        DocumentDataset(DATA / "train.csv", tokenizer, label2id, MAX_LEN),
        batch_size=BATCH, shuffle=True,
    )
    val_loader = DataLoader(
        DocumentDataset(DATA / "val.csv", tokenizer, label2id, MAX_LEN), batch_size=BATCH
    )
    test_loader = DataLoader(
        DocumentDataset(DATA / "test.csv", tokenizer, label2id, MAX_LEN), batch_size=BATCH
    )

    opt = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    save_label_maps(label2id, MODEL_DIR / "label2id.json")

    history, best_f1 = [], -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            opt.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            out.loss.backward()
            opt.step()
            sched.step()
            loss_sum += out.loss.item()

        val = evaluate(model, val_loader, device, id2label)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / len(train_loader),
            "val_accuracy": val["accuracy"],
            "val_f1_macro": val["f1_macro"],
        }
        history.append(row)
        print(
            f"Epoch {epoch}: loss={row['train_loss']:.4f} "
            f"val_acc={row['val_accuracy']:.4f} val_f1={row['val_f1_macro']:.4f}"
        )

        if val["f1_macro"] > best_f1:
            best_f1 = val["f1_macro"]
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"  saved best checkpoint (val F1={best_f1:.4f})")

    (RESULTS / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    test = evaluate(model, test_loader, device, id2label)
    labels = [id2label[i] for i in range(len(id2label))]

    summary = {
        "model": "distilbert-base-uncased",
        "epochs": EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH,
        "max_length": MAX_LEN,
        "best_val_f1_macro": best_f1,
        "test_accuracy": test["accuracy"],
        "test_f1_macro": test["f1_macro"],
        "test_f1_weighted": test["f1_weighted"],
        "per_class": test["report"],
        "label2id": label2id,
    }
    (RESULTS / "test_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_cm(test["y_true"], test["y_pred"], labels, RESULTS / "confusion_matrix.png")

    print("\n=== TEST ===")
    print(f"Accuracy:  {test['accuracy']:.4f}")
    print(f"F1 macro:  {test['f1_macro']:.4f}")
    print(f"Model:     {MODEL_DIR}")
    print(f"Results:   {RESULTS}")


if __name__ == "__main__":
    main()
