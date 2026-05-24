"""Verify manual DistilBERT files before training."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "models" / "pretrained_distilbert"

REQUIRED = [
    "config.json",
    "vocab.txt",
    "tokenizer_config.json",
    "tokenizer.json",
    "pytorch_model.bin",
]

if __name__ == "__main__":
  print(f"Checking {DIR}\n")
  ok = True
  for name in REQUIRED:
    p = DIR / name
    if not p.exists():
      print(f"  MISSING: {name}")
      ok = False
    else:
      mb = p.stat().st_size / 1_000_000
      print(f"  OK: {name} ({mb:.1f} MB)")
      if name == "pytorch_model.bin" and mb < 250:
        print("       ^ too small — re-download pytorch_model.bin (~268 MB)")
        ok = False
  if ok:
    print("\nAll set. Run: python train_document_classifier.py")
  else:
    print("\nSee models/pretrained_distilbert/README.md for download links.")
    raise SystemExit(1)
