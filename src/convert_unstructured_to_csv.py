from pathlib import Path
from typing import List
import pandas as pd

RAW_ROOT = Path("data/raw")
UNSTRUCTURED_DIR = RAW_ROOT / "unstructured"
OUTPUT_DIR = RAW_ROOT  # final CSVs here so simple_loader.py can pick them up

def collect_lines_from_file(path: Path) -> List[str]:
    """Read a text/markdown file and return non-empty lines."""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f.readlines()]
    return [ln for ln in lines if ln]

def build_dataset(label: str, mode: str = "line") -> None:
    """
    Convert all .txt/.md under data/raw/unstructured/<label>/ to one CSV.

    mode = "line"  -> each non-empty line is one sample
    mode = "file"  -> each whole file is one sample
    """
    label_dir = UNSTRUCTURED_DIR / label
    if not label_dir.exists():
        raise FileNotFoundError(f"Folder not found: {label_dir}")

    texts: List[str] = []

    for path in label_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".md"}:
            continue

        if mode == "line":
            texts.extend(collect_lines_from_file(path))
        else:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
            if content:
                texts.append(content)

    if not texts:
        print(f"[WARN] No text found in {label_dir}")
        return

    df = pd.DataFrame({"text": texts, "sentiment": label})
    out_path = OUTPUT_DIR / f"unstructured_{label}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Saved {len(df)} rows to {out_path}")

def main() -> None:
    """
    Example usage:

    Put files under:
      data/raw/unstructured/positive/...
      data/raw/unstructured/negative/...
      data/raw/unstructured/neutral/...

    Then run this script to create labeled CSVs for each label,
    which simple_loader.py can merge like any other dataset.
    """
    labels = ["positive", "negative", "neutral"]  # adjust to folders you create

    for lbl in labels:
        try:
            build_dataset(label=lbl, mode="line")
        except FileNotFoundError:
            print(f"[INFO] Skipping label '{lbl}' (folder missing).")

if __name__ == "__main__":
    main()
