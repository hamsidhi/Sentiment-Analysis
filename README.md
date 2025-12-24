# Sentiment Analysis Project 🎯

A **multilingual sentiment analysis system** trained on ~1.4M samples...

![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg) 
![Status](https://img.shields.io/badge/status-production-brightgreen.svg)
![Accuracy](https://img.shields.io/badge/accuracy-78.2%25-orange.svg)

**Educational -  Practical -  Extensible -  Production-Ready**

✅ **Complete pipeline**: Data → Preprocessing → Training → Prediction  
✅ **Multiple models**: TF-IDF + Logistic Regression (78.2% accuracy)  
✅ **Multilingual**: English + Turkish (extensible to any language)  
✅ **Universal import**: Hugging Face, Kaggle, local files, URLs  
✅ **Production-ready**: Model versioning, metadata, logging  
✅ **Well-documented**: Professional rules & guidelines included  

## 📥 Full Datasets (1.4M Samples)

**Samples included** (`data/raw/*_sample.csv` - 100 rows each for testing)

**Full datasets** (auto-download with one command):
'''
python src/import_datasets.py # Downloads ALL 1.4M samples
'''

## 📊 Dataset Overview

| Dataset | Source | Size | Command |
|---------|--------|------|---------|
| Amazon Reviews | Local sample | 5K full | `import_datasets.py` |
| IMDB Movies | [HuggingFace](https://huggingface.co/datasets/stanfordnlp/imdb) | 50K | Auto |
| Turkish Sentiment | [HF Turkish](https://huggingface.co/datasets/maydogan/Turkish_SentimentAnalysis_TRSAv1) | 150K | Auto |
| Twitter | [Sentiment140](http://help.sentiment140.com/for-students) | 100K | Auto |


## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/hamsidhi/Sentiment-Analysis.git
cd Sentiment-Analysis
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Import Datasets (One Command!)
```bash
# Auto-downloads from Hugging Face, Kaggle, etc.
python src/import_datasets.py
```

### 3. Train Model
```bash
python src/train_baseline.py
# Output: models/baseline_tfidf_logreg.pkl (78.2% accuracy)
```

### 4. Make Predictions
```bash
# Interactive CLI
python src/predict_example.py
```
```
Enter text: This product is amazing!
→ positive (confidence: 92.3%)
Enter text: Bu film çok kötü idi. (Turkish)
→ negative (confidence: 87.1%)
```

**Python API:**
```python
import joblib
model = joblib.load("models/baseline_tfidf_logreg.pkl")
result = model.predict(["I love this!", "Terrible service"])
# [1, 0] → positive, negative
```

## 📁 Project Structure

```
sentiment-analysis-project/
├── data/
│   ├── raw/          # Downloaded datasets (ignored by git)
│   └── processed/    # Combined 1.4M samples (ignored)
├── src/              # 🚀 Source code (UPLOADED)
│   ├── simple_loader.py
│   ├── train_baseline.py
│   ├── predict_example.py
│   └── import_datasets.py
├── models/           # Trained models (ignored)
├── reports/          # Visualizations (ignored)
├── rules_files/      # 📚 Development standards (UPLOADED)
├── requirements.txt  # Dependencies
├── README.md         # This file
└── .gitignore        # Clean repo only
```

**Uploaded to GitHub**: ~500KB code/docs  
**Stays local**: ~2.5GB data/models (via .gitignore)

## 🔧 Features

### ✅ Universal Data Import
```
1. Hugging Face: maydogan/Turkish_SentimentAnalysis_TRSAv1
2. Kaggle: Any dataset
3. Local CSV/JSON/Parquet
4. Direct URL download
```
**Auto-detects** text/sentiment columns!

### ✅ Production Model (78.2% Accuracy)
```
Test Set (240K samples):
precision    recall  f1-score   support
Negative      0.771    0.758     0.765   112592
Positive      0.790    0.803     0.797   128104
accuracy                          0.782   240696
```

**Architecture:**
```
Input Text → TF-IDF (50K features, bigrams) → Logistic Regression (balanced)
```

### ✅ Comprehensive Analytics
- 6 individual dataset reports + 1 combined
- Text length distributions
- Sentiment balance charts
- Sample text analysis

## 📥 One-Command Dataset Import

```bash
python src/import_datasets.py
```
```
1. Hugging Face Dataset
2. Kaggle Dataset
3. Local File
4. Remote URL
5. Exit

Enter choice: 1
Dataset ID: maydogan/Turkish_SentimentAnalysis_TRSAv1
Split: train
✓ Saved: data/raw/huggingface_maydogan_*.csv
```

## 🎓 Development Standards

**Professional rules included** (`rules_files/`):

| Document | Purpose |
|----------|---------|
| **PROJECT_RULES.md** | Structure & organization |
| **CODE_QUALITY_RULES.md** | Python standards (type hints, docstrings) |
| **DATA_MODEL_RULES.md** | Data/ML best practices |
| **TESTING_DEPLOYMENT_RULES.md** | Testing (80%+ coverage) & deployment |
| **QUICK_REFERENCE.md** | Daily checklists |

## 🔄 Full Pipeline (5 Minutes)

```bash
# 1. Download datasets
python src/import_datasets.py

# 2. Combine (1.4M rows)
python src/simple_loader.py

# 3. Visualize
python src/visualize_datasets.py

# 4. Train (78.2% accuracy)
python src/train_baseline.py

# 5. Predict
python src/predict_example.py
```

## 📈 Model Performance

```
TF-IDF + Logistic Regression (1.4M samples):
                precision    recall  f1-score   support
Negative         0.771     0.758     0.765    112592
Positive         0.790     0.803     0.797    128104
accuracy                             0.782    240696
```

## 🤝 Contributing

1. **Read** `rules_files/CODE_QUALITY_RULES.md`
2. **Follow** standards (snake_case, type hints, <50 line functions)
3. **Add** tests (80%+ coverage)
4. **Branch**: `git checkout -b feature/your-feature`
5. **Commit**: `git commit -m "[Feature] Add BERT support"`

## 📦 Dependencies

```bash
pip install -r requirements.txt
```
**Core**: pandas, scikit-learn, matplotlib, seaborn  
**Data**: datasets, huggingface-hub, kaggle  
**Models**: joblib, tqdm

## 🔐 License

[MIT License](LICENSE) - Free for commercial use

## 👤 Author

**Hamza Siddiqui**  
Data Science Student | Atharva College, Malad, Mumbai

***

**Last Updated**: December 2025  
**Samples**: 1.4M (English + Turkish)  
**Accuracy**: 78.2%  
**Status**: Production-ready  

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/hamsidhi/Sentiment-Analysis/issues)
- **Rules**: `rules_files/README_RULES.md`
- **Quick Start**: See commands above

***

**Ready to analyze sentiment in multiple languages!** 🌍🎉
