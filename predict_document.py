from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent / "models" / "document_classifier"


def predict(text: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    with torch.no_grad():
        pred_id = int(model(**enc).logits.argmax(-1).item())
    return model.config.id2label[str(pred_id)]


if __name__ == "__main__":
    print("invoice  ->", predict("INVOICE\nInvoice #: INV-1\nTotal Due: $100"))
    print("t4       ->", predict("T4 Statement of Remuneration Paid\nEmployment Income: $50,000"))
