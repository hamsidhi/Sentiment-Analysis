"""
Universal dataset importer for Hugging Face, Kaggle, and local files.

Supports:
  - Hugging Face datasets (public & private)
  - Kaggle datasets
  - Local CSV/JSON/Parquet files
  - Remote URLs (direct downloads)

All data saved to data/raw/ following project structure rules.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

import pandas as pd
import requests

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING SETUP (per CODE_QUALITY_RULES.md)
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

RAW_DATA_DIR = Path("data/raw")
METADATA_TEMPLATE = {
    "source": None,
    "source_type": None,  # huggingface, kaggle, local, url
    "download_date": None,
    "rows": None,
    "columns": None,
    "text_column": None,
    "sentiment_column": None,
    "processing_notes": None,
}


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY: Detect and rename text/sentiment columns
# ═══════════════════════════════════════════════════════════════════════════

def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    """
    Auto-detect text column from common names.

    Returns:
    --------
    str or None: Column name if found, else None
    """
    text_keywords = {"review", "text", "tweet", "comment", "content", "sentence", "body"}
    for col in df.columns:
        if col.lower() in text_keywords or any(
            kw in col.lower() for kw in text_keywords
        ):
            return col
    return None


def detect_sentiment_column(df: pd.DataFrame) -> Optional[str]:
    """
    Auto-detect sentiment/label column from common names.

    Returns:
    --------
    str or None: Column name if found, else None
    """
    sentiment_keywords = {
        "sentiment", "label", "class", "target", "rating", "score",
        "polarity", "category"
    }
    for col in df.columns:
        if col.lower() in sentiment_keywords or any(
            kw in col.lower() for kw in sentiment_keywords
        ):
            return col
    return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename detected columns to standard 'text' and 'sentiment'.

    Follows DATA_MODEL_RULES.md standardization.
    """
    df = df.copy()
    
    text_col = detect_text_column(df)
    if text_col and text_col != "text":
        df = df.rename(columns={text_col: "text"})
        logger.info(f"Renamed '{text_col}' -> 'text'")
    
    sentiment_col = detect_sentiment_column(df)
    if sentiment_col and sentiment_col != "sentiment":
        df = df.rename(columns={sentiment_col: "sentiment"})
        logger.info(f"Renamed '{sentiment_col}' -> 'sentiment'")
    
    return df


# ═══════════════════════════════════════════════════════════════════════════
# IMPORTER: Hugging Face Datasets
# ═══════════════════════════════════════════════════════════════════════════

def import_huggingface(
    dataset_name: str,
    split: str = "train",
    subset: Optional[str] = None,
    output_name: Optional[str] = None,
) -> Path:
    """
    Import dataset from Hugging Face Hub.

    Parameters:
    -----------
    dataset_name : str
        Full dataset identifier (e.g., "maydogan/Turkish_SentimentAnalysis_TRSAv1")
    split : str
        Dataset split to load (e.g., "train", "test", "validation")
    subset : str, optional
        Config/subset name if dataset has multiple (e.g., "tr" for Turkish)
    output_name : str, optional
        Custom output filename. Default: dataset_name with date

    Returns:
    --------
    Path: Path to saved CSV in data/raw/

    Example:
    --------
    >>> path = import_huggingface(
    ...     "maydogan/Turkish_SentimentAnalysis_TRSAv1",
    ...     split="train"
    ... )
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install: pip install datasets")

    logger.info(f"Loading Hugging Face dataset: {dataset_name} (split={split})")

    # Load dataset
    try:
        if subset:
            dataset = load_dataset(dataset_name, subset, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
    except Exception as exc:
        logger.error(f"Failed to load dataset: {exc}")
        raise

    # Convert to DataFrame
    df = dataset.to_pandas()
    logger.info(f"Loaded {len(df):,} rows from Hugging Face")

    # Standardize columns
    df = standardize_columns(df)

    # Save to data/raw/
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_name = output_name or f"huggingface_{dataset_name.replace('/', '_')}"
    output_path = RAW_DATA_DIR / f"{output_name}.csv"

    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Saved to {output_path}")

    # Save metadata
    metadata = METADATA_TEMPLATE.copy()
    metadata.update({
        "source": dataset_name,
        "source_type": "huggingface",
        "download_date": datetime.now().isoformat(),
        "rows": len(df),
        "columns": list(df.columns),
        "text_column": "text" if "text" in df.columns else None,
        "sentiment_column": "sentiment" if "sentiment" in df.columns else None,
    })
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# IMPORTER: Kaggle Datasets
# ═══════════════════════════════════════════════════════════════════════════

def import_kaggle(
    dataset_id: str,
    output_name: Optional[str] = None,
) -> Path:
    """
    Import dataset from Kaggle.

    Requires Kaggle API setup:
      1. Download API key from https://www.kaggle.com/settings/account
      2. Place at ~/.kaggle/kaggle.json
      3. chmod 600 ~/.kaggle/kaggle.json

    Parameters:
    -----------
    dataset_id : str
        Kaggle dataset identifier (e.g., "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    output_name : str, optional
        Custom output filename

    Returns:
    --------
    Path: Path to saved CSV in data/raw/

    Example:
    --------
    >>> path = import_kaggle(
    ...     "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
    ... )
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError("Install: pip install kaggle")

    logger.info(f"Downloading Kaggle dataset: {dataset_id}")

    # Initialize Kaggle API
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        logger.error(
            "Kaggle authentication failed. "
            "Setup: https://www.kaggle.com/settings/account -> API -> Download"
        )
        raise

    # Download to temp folder
    temp_dir = RAW_DATA_DIR / ".kaggle_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        api.dataset_download_files(dataset_id, path=temp_dir, unzip=True)
        logger.info(f"Downloaded Kaggle dataset to {temp_dir}")
    except Exception as exc:
        logger.error(f"Download failed: {exc}")
        raise

    # Find CSV files
    csv_files = list(temp_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {temp_dir}")

    # Use largest CSV (usually the main dataset)
    csv_file = max(csv_files, key=lambda p: p.stat().st_size)
    logger.info(f"Using CSV: {csv_file.name}")

    # Load and standardize
    df = pd.read_csv(csv_file, low_memory=False)
    df = standardize_columns(df)

    # Save to data/raw/
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_name = output_name or f"kaggle_{dataset_id.split('/')[-1]}"
    output_path = RAW_DATA_DIR / f"{output_name}.csv"

    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Saved to {output_path}")

    # Save metadata
    metadata = METADATA_TEMPLATE.copy()
    metadata.update({
        "source": dataset_id,
        "source_type": "kaggle",
        "download_date": datetime.now().isoformat(),
        "rows": len(df),
        "columns": list(df.columns),
        "text_column": "text" if "text" in df.columns else None,
        "sentiment_column": "sentiment" if "sentiment" in df.columns else None,
    })
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Cleanup temp
    import shutil
    shutil.rmtree(temp_dir)

    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# IMPORTER: Local Files
# ═══════════════════════════════════════════════════════════════════════════

def import_local(
    file_path: Union[str, Path],
    output_name: Optional[str] = None,
) -> Path:
    """
    Import dataset from local CSV/JSON/Parquet file.

    Parameters:
    -----------
    file_path : str or Path
        Path to local file (CSV, JSON, or Parquet)
    output_name : str, optional
        Custom output filename

    Returns:
    --------
    Path: Path to saved CSV in data/raw/

    Example:
    --------
    >>> path = import_local("~/Downloads/reviews.csv")
    """
    file_path = Path(file_path).expanduser()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Loading local file: {file_path}")

    # Load based on extension
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".csv":
            df = pd.read_csv(file_path, low_memory=False)
        elif suffix == ".json":
            df = pd.read_json(file_path)
        elif suffix == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
    except Exception as exc:
        logger.error(f"Failed to load file: {exc}")
        raise

    logger.info(f"Loaded {len(df):,} rows")

    # Standardize columns
    df = standardize_columns(df)

    # Save to data/raw/
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_name = output_name or file_path.stem
    output_path = RAW_DATA_DIR / f"{output_name}.csv"

    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Saved to {output_path}")

    # Save metadata
    metadata = METADATA_TEMPLATE.copy()
    metadata.update({
        "source": str(file_path),
        "source_type": "local",
        "download_date": datetime.now().isoformat(),
        "rows": len(df),
        "columns": list(df.columns),
        "text_column": "text" if "text" in df.columns else None,
        "sentiment_column": "sentiment" if "sentiment" in df.columns else None,
    })
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# IMPORTER: Remote URLs
# ═══════════════════════════════════════════════════════════════════════════

def import_from_url(
    url: str,
    output_name: Optional[str] = None,
) -> Path:
    """
    Download and import dataset from a direct URL.

    Supports CSV, JSON, Parquet files.

    Parameters:
    -----------
    url : str
        Direct download URL
    output_name : str, optional
        Custom output filename

    Returns:
    --------
    Path: Path to saved CSV in data/raw/

    Example:
    --------
    >>> path = import_from_url(
    ...     "https://example.com/reviews.csv"
    ... )
    """
    logger.info(f"Downloading from URL: {url}")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except Exception as exc:
        logger.error(f"Download failed: {exc}")
        raise

    # Detect format from URL
    url_path = url.split("?")[0]  # Remove query params
    suffix = Path(url_path).suffix.lower()

    # Write to temp file and load
    temp_file = Path("temp_download") / Path(url_path).name
    temp_file.parent.mkdir(parents=True, exist_ok=True)

    with open(temp_file, "wb") as f:
        f.write(response.content)

    logger.info(f"Downloaded {temp_file.stat().st_size / 1024 / 1024:.1f} MB")

    # Load based on format
    try:
        if suffix == ".csv":
            df = pd.read_csv(temp_file, low_memory=False)
        elif suffix == ".json":
            df = pd.read_json(temp_file)
        elif suffix == ".parquet":
            df = pd.read_parquet(temp_file)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
    finally:
        temp_file.unlink()

    logger.info(f"Loaded {len(df):,} rows")

    # Standardize columns
    df = standardize_columns(df)

    # Save to data/raw/
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_name = output_name or Path(url_path).stem
    output_path = RAW_DATA_DIR / f"{output_name}.csv"

    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Saved to {output_path}")

    # Save metadata
    metadata = METADATA_TEMPLATE.copy()
    metadata.update({
        "source": url,
        "source_type": "url",
        "download_date": datetime.now().isoformat(),
        "rows": len(df),
        "columns": list(df.columns),
        "text_column": "text" if "text" in df.columns else None,
        "sentiment_column": "sentiment" if "sentiment" in df.columns else None,
    })
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# MAIN: Interactive importer
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Interactive menu for importing datasets.

    Usage:
    ------
    python src/import_datasets.py
    """
    print("\n" + "=" * 70)
    print("UNIVERSAL DATASET IMPORTER")
    print("=" * 70)
    print("\nChoose import source:")
    print("  1. Hugging Face Dataset")
    print("  2. Kaggle Dataset")
    print("  3. Local File (CSV/JSON/Parquet)")
    print("  4. Remote URL")
    print("  5. Exit")
    print()

    choice = input("Enter choice (1-5): ").strip()

    if choice == "1":
        dataset_id = input("Enter Hugging Face dataset ID (e.g., maydogan/Turkish_SentimentAnalysis_TRSAv1): ").strip()
        split = input("Enter split [train]: ").strip() or "train"
        subset = input("Enter subset/config (optional, press Enter to skip): ").strip() or None
        output = input("Custom output filename (optional, press Enter for default): ").strip() or None

        path = import_huggingface(dataset_id, split=split, subset=subset, output_name=output)
        print(f"\n✓ Successfully imported to {path}")

    elif choice == "2":
        dataset_id = input("Enter Kaggle dataset ID (e.g., lakshmi25npathi/imdb-dataset-of-50k-movie-reviews): ").strip()
        output = input("Custom output filename (optional): ").strip() or None

        path = import_kaggle(dataset_id, output_name=output)
        print(f"\n✓ Successfully imported to {path}")

    elif choice == "3":
        file_path = input("Enter local file path: ").strip()
        output = input("Custom output filename (optional): ").strip() or None

        path = import_local(file_path, output_name=output)
        print(f"\n✓ Successfully imported to {path}")

    elif choice == "4":
        url = input("Enter URL to CSV/JSON/Parquet file: ").strip()
        output = input("Custom output filename (optional): ").strip() or None

        path = import_from_url(url, output_name=output)
        print(f"\n✓ Successfully imported to {path}")

    elif choice == "5":
        print("Exiting.")
        return
    else:
        print("Invalid choice. Exiting.")
        return

    print("\nNext steps:")
    print("  1. python src/simple_loader.py    (combine all datasets)")
    print("  2. python src/visualize_datasets.py (create charts)")
    print("  3. python src/train_baseline.py     (train model)")
    print()


if __name__ == "__main__":
    main()
