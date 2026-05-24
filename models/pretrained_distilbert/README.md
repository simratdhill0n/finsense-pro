# Manual DistilBERT download

Download these files from Hugging Face and put them **in this folder** (`models/pretrained_distilbert/`):

**Page:** https://huggingface.co/distilbert-base-uncased/tree/main

| File | Approx. size |
|------|----------------|
| `config.json` | small |
| `vocab.txt` | small |
| `tokenizer_config.json` | small |
| `tokenizer.json` | ~0.5 MB |
| `pytorch_model.bin` | **~268 MB** |

## Steps

1. Open the link above in your browser.
2. Click each file → download.
3. Save all 5 files into this folder (same level as this README).
4. Confirm `pytorch_model.bin` is about **250+ MB** (not a partial download).
5. From the project root, run:
   ```bash
   python scripts/setup_data.py
   python scripts/create_splits.py
   python train_document_classifier.py
   ```

## Optional: Hugging Face CLI

If you have space on disk and the CLI installed:

```bash
pip install huggingface_hub
huggingface-cli download distilbert-base-uncased --local-dir models/pretrained_distilbert
```
