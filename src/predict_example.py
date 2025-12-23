from pathlib import Path
import joblib

MODEL_PATH = Path("models/baseline_tfidf_logreg.pkl")

model = joblib.load(MODEL_PATH)

while True:
    text = input("\nEnter text (or 'quit'): ").strip()
    if text.lower() == "quit":
        break
    pred = model.predict([text])[0]
    label = "positive" if pred == 1 else "negative"
    print("Sentiment:", label)
