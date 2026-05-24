"""70% train / 15% val / 15% test (stratified by document_type)."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 42

if __name__ == "__main__":
    df = pd.read_csv(DATA / "invoices_dataset.csv")
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["document_type"], random_state=SEED
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["document_type"], random_state=SEED
    )
    train.to_csv(DATA / "train.csv", index=False)
    val.to_csv(DATA / "val.csv", index=False)
    test.to_csv(DATA / "test.csv", index=False)
    print(f"train={len(train):,}  val={len(val):,}  test={len(test):,}")
