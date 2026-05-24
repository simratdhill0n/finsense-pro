"""Copy teammate CSV into data/invoices_dataset.csv if needed."""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEST = DATA / "invoices_dataset.csv"

SOURCES = [
    Path(r"d:\Everything unwanted\Downloadsss\invoices_dataset (1).csv"),
    Path(r"d:\Everything unwanted\Downloadsss\invoices_dataset.csv"),
]

if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"Already exists: {DEST}")
    else:
        for src in SOURCES:
            if src.exists():
                shutil.copy2(src, DEST)
                print(f"Copied: {src}")
                break
        else:
            raise FileNotFoundError(
                "Place invoices_dataset.csv in data/ "
                "(or put invoices_dataset (1).csv in Downloads)"
            )

    import pandas as pd

    df = pd.read_csv(DEST)
    print(f"Rows: {len(df):,}")
    print(df["document_type"].value_counts())
