import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

DATA_PATH = Path("data/processed/sentiment_data.csv")

def load_and_filter():
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Keep only rows with sentiment
    if "sentiment" not in df.columns:
        raise ValueError("No 'sentiment' column in processed data.")

    df = df.dropna(subset=["text", "sentiment"])
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()

    # Universal mapping to binary labels
    positive_vals = {"positive", "pos", "1", "5.0", "4.0"}
    negative_vals = {"negative", "neg", "0", "1.0", "2.0"}

    def map_label(x):
        if x in positive_vals:
            return 1
        if x in negative_vals:
            return 0
        # For numeric stars like 3.0 etc.
        try:
            v = float(x)
            return int(v >= 3.0)
        except Exception:
            return None

    df["label"] = df["sentiment"].map(map_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    return df[["text", "label"]]

def train_baseline():
    print("[INFO] Loading processed data...")
    df = load_and_filter()
    print(f"[INFO] Using {len(df):,} samples for training/evaluation")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # Universal TF‑IDF + Logistic Regression pipeline
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                stop_words="english"
            )),
            ("clf", LogisticRegression(
                max_iter=200,
                n_jobs=-1,
                class_weight="balanced"
            )),
        ]
    )

    print("[INFO] Training model...")
    model.fit(X_train, y_train)

    print("[INFO] Evaluating on test set...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))

    # Save model for reuse
    Path("models").mkdir(exist_ok=True)
    import joblib
    joblib.dump(model, "models/baseline_tfidf_logreg.pkl")
    print("[INFO] Saved model to models/baseline_tfidf_logreg.pkl")

if __name__ == "__main__":
    train_baseline()
