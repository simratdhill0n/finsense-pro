# FinSense Pro

**Privacy-Preserving Invoice Fraud Detection System**  
**AASD 4014 - Deep Learning I | George Brown Polytechnic**

### Team
- **Project Manager**: Simrat Pal Singh Dhillon (101598631)
- Rahul Verma (101525530)
- Darundeep Verma (101448044)
- Misra Erol (101638309)

### Project Overview
FinSense Pro is a deep learning system that detects fraudulent invoices for Canadian small businesses using:
- 4 PyTorch deep learning models
- Federated Learning (Flower simulation)
- Full EDA, hyperparameter tuning, and evaluation

**Models:**
1. Transformer Document Classifier
2. BERT-style Entity Extractor
3. LSTM + Autoencoder Anomaly Detector
4. Seq2Seq Explanation Generator

---

## Task 2 — Document classifier (Rahul) — DistilBERT

Classifies `invoice_text` into: **invoice**, **t4**, **bank_statement**, **expense_report**.

### Step 1 — Install

```bash
pip install -r requirements.txt
```

### Step 2 — Download DistilBERT **manually** (recommended)

Open: **https://huggingface.co/distilbert-base-uncased/tree/main**

Download these 5 files into `models/pretrained_distilbert/`:

- `config.json`
- `vocab.txt`
- `tokenizer_config.json`
- `tokenizer.json`
- `pytorch_model.bin` (~268 MB)

More detail: `models/pretrained_distilbert/README.md`

Check files:

```bash
python scripts/check_distilbert.py
```

### Step 3 — Data & splits

```bash
python data_generation_pipeline.py
python scripts/create_splits.py
```

### Step 4 — Train

```bash
python train_document_classifier.py
python predict_document.py
```

### Outputs

- `models/document_classifier/` — fine-tuned model
- `results/task2/test_metrics.json` — accuracy, F1
- `results/task2/confusion_matrix.png`

---

## Task 1 — Data generation

```bash
python data_generation_pipeline.py
```

Generates `data/invoices_dataset.csv` with 4 document types and fraud anomaly labels.
