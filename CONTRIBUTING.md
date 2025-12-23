```markdown
# Contributing to Sentiment Analysis Project 🚀

Thank you for your interest in contributing! This document outlines how to contribute effectively to our multilingual sentiment analysis system (1.4M samples, 78.2% accuracy).

## 📋 Before You Start

### 1. Read the Documentation
```
✅ README.md                 - Project overview & quick start
✅ rules_files/              - Professional development standards
   ├── PROJECT_RULES.md      - Project structure
   ├── CODE_QUALITY_RULES.md - Python coding standards
   ├── DATA_MODEL_RULES.md   - Data/ML best practices
   └── TESTING_DEPLOYMENT_RULES.md - Testing & deployment
✅ CONTRIBUTING.md           - This file (you're reading it!)
```

### 2. Setup Your Environment
```
git clone https://github.com/hamsidhi/Sentiment-Analysis.git
cd Sentiment-Analysis
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Verify Setup
```
python src/predict_example.py  # Should run without errors
pytest tests/                 # Should pass all tests
```

## 🔧 Development Workflow

### Step 1: Create a Feature Branch
```
# From main branch
git checkout main
git pull origin main

# Create your branch
git checkout -b feature/add-bert-model
# or
git checkout -b bugfix/fix-turkish-encoding
# or
git checkout -b docs/improve-readme
```

### Step 2: Follow Code Standards
**Mandatory** (from `rules_files/CODE_QUALITY_RULES.md`):

| Rule | Example |
|------|---------|
| **Naming** | `snake_case` vars, `PascalCase` classes, `UPPER_CASE` constants |
| **Type hints** | `def process_data(df: pd.DataFrame) -> pd.DataFrame:` |
| **Docstrings** | Google style with Parameters/Returns |
| **Functions** | `< 50 lines each` |
| **Logging** | `logger.info()` not `print()` |
| **No hardcoding** | Use `Path` from pathlib |

**Code Template:**
```
import logging
from typing import List
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

def process_sentiment_data(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "sentiment"
) -> pd.DataFrame:
    """
    Process sentiment data with validation and cleaning.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with text and sentiment columns
    text_col : str
        Name of text column (default: 'text')
    label_col : str
        Name of sentiment column (default: 'sentiment')
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe ready for training
        
    Raises:
    -------
    ValueError
        If required columns are missing
    """
    if text_col not in df.columns or label_col not in df.columns:
        logger.error(f"Missing columns: {text_col}, {label_col}")
        raise ValueError("Required columns missing")
    
    logger.info(f"Processing {len(df)} rows")
    df = df[[text_col, label_col]].copy()
    df = df.dropna()
    
    logger.info(f"After cleaning: {len(df)} rows")
    return df
```

### Step 3: Write Tests (80%+ Coverage Required)
**Create tests in `tests/` following this template:**

```
import pytest
import pandas as pd
from src.your_module import process_sentiment_data

class TestYourFunction:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            "text": ["good", "bad", ""],
            "sentiment": [1, 0, None]
        })
    
    def test_normal_case(self, sample_data):
        """Test normal input."""
        result = process_sentiment_data(sample_data)
        assert len(result) == 2
        assert result["sentiment"].isin().all()[1]
    
    def test_missing_columns(self):
        """Test missing column error."""
        df = pd.DataFrame({"wrong": ["test"]})
        with pytest.raises(ValueError):
            process_sentiment_data(df)
    
    def test_empty_dataframe(self):
        """Test empty input."""
        result = process_sentiment_data(pd.DataFrame())
        assert result.empty
```

**Run tests:**
```
pytest tests/ --cov=src --cov-report=html  # Target: 80%+ coverage
```

### Step 4: Commit with Conventional Messages
```
Format: [Type] Short description (max 50 chars)

Types:
[Feature] New functionality
[Bugfix] Fix a bug
[Docs] Documentation changes
[Maintenance] Refactoring, cleanup
[Tests] Add/fix tests
[Performance] Performance improvements
```

**Examples:**
```
git commit -m "[Feature] Add BERT model support"
git commit -m "[Bugfix] Fix Turkish UTF-8 encoding issue"
git commit -m "[Docs] Update README with new examples"
git commit -m "[Tests] Add 80% test coverage for preprocessor"
git commit -m "[Maintenance] Clean up unused imports"
```

### Step 5: Push & Create Pull Request
```
git push origin feature/your-feature-branch
```

**Create PR on GitHub with:**
1. **Clear title**: `[Feature] Add BERT model support`
2. **Description**:
   ```
   ## What
   Added BERT transformer model alongside TF-IDF baseline
   
   ## Why
   Improves accuracy from 78.2% → 84.5% on test set
   
   ## Changes
   - New `train_bert.py`
   - Updated `import_datasets.py`
   - Tests: 85% coverage
   
   ## Results
   | Model | Accuracy |
   |-------|----------|
   | TF-IDF | 78.2% |
   | BERT   | 84.5% |
   ```
3. **Checklist**:
   ```
   - [x] Follows CODE_QUALITY_RULES.md
   - [x] Tests pass (80%+ coverage)
   - [x] No linting errors
   - [x] Documentation updated
   ```

## 📊 Types of Contributions

### 🌱 Beginner-Friendly
```
- [ ] Fix typos in docs
- [ ] Add example datasets
- [ ] Improve README sections
- [ ] Add more preprocessing options
- [ ] Create tutorial notebooks
```

### 🔧 Intermediate
```
- [ ] Add new ML models (XGBoost, LightGBM)
- [ ] Implement hyperparameter tuning
- [ ] Add model ensemble methods
- [ ] Create Streamlit/Flask demo
- [ ] Add CI/CD GitHub Actions
```

### 🚀 Advanced
```
- [ ] Multi-language BERT support
- [ ] Active learning implementation
- [ ] Model interpretability (SHAP/LIME)
- [ ] REST API with FastAPI
- [ ] Docker deployment
```

## 🧪 Testing Requirements

### Coverage Targets
```
src/          : 80%+
src/data/     : 85%+
src/models/   : 90%+
src/utils/    : 75%+
Overall       : 80%+
```

### Test Checklist
```
□ Unit tests for all public functions
□ Edge cases (empty, None, extreme values)
□ Error conditions (raises expected exceptions)
□ Different input types (str, list, DataFrame)
□ pytest --cov-report=html (visual verification)
```

## 📝 Commit Message Guidelines

### Good Examples
```
[Feature] Add Turkish language support
[Bugfix] Fix NaN handling in preprocessor
[Docs] Add TF-IDF explanation to README
[Tests] Achieve 85% coverage for data module
[Performance] Speed up TF-IDF by 3x with parallel
```

### Bad Examples
```
fix                     ❌ Too vague
update readme           ❌ No type prefix
Added cool new feature  ❌ Capitalization, too long
```

## 🐛 Bug Report Template

**Use this format for bug reports:**

```
## Bug Report

### Environment
- OS: [Windows/Linux/macOS]
- Python: [3.12]
- Commit: [abc1234]

### Steps to Reproduce
1. Run `python src/train_baseline.py`
2. ...
3. ...

### Expected Behavior
Model trains successfully with 78.2% accuracy

### Actual Behavior
```
Traceback (most recent call last):
  File "...", line 42, in <module>
    ...
ValueError: ...
```

### Dataset Used
- [x] IMDB
- [ ] Turkish HF
- [ ] Custom dataset

### Additional Context
[Any screenshots, logs, etc.]
```

## 💡 Feature Request Template

```
## Feature Request

### Problem
[Describe the problem this feature solves]

### Proposed Solution
[Describe your solution]

### Alternatives Considered
[Other approaches you thought about]

### Example Usage
```python
# How would users use this feature?
new_feature_example()
```

### Benefits
- [ ] Improves accuracy
- [ ] Adds new dataset support
- [ ] Better UX
- [ ] Production readiness
```

## 🔍 Code Review Checklist

**Before submitting PR, verify:**

```
CODE QUALITY
□ snake_case variables, PascalCase classes
□ Type hints on all functions
□ Google-style docstrings
□ Functions < 50 lines
□ No print() statements (use logging)
□ No hardcoded paths

TESTING
□ 80%+ coverage (pytest --cov=src)
□ Edge cases tested
□ Error conditions raise exceptions
□ All tests pass: pytest tests/ -v

DATA
□ raw/ data not modified
□ Metadata JSON saved
□ Processing steps logged

MODEL
□ Random seeds set (reproducibility)
□ Train/val/test split (70/15/15)
□ Metrics logged with metadata
□ Model saved with versioning
```

## 🚀 Release Process

```
1. Update version in README, setup.py
2. Run full test suite: pytest tests/ --cov=src
3. Tag release: git tag -a v1.2.0 -m "Release v1.2.0"
4. Push tags: git push origin v1.2.0
5. Create GitHub Release with changelog
```

## ❓ Frequently Asked Questions

### Q: Where do new files go?
```
src/              → Main Python code
tests/            → Unit tests
data/raw/         → New datasets (ignored by git)
data/processed/   → Processed data (ignored)
models/           → Trained models (ignored)
rules_files/      → Documentation/rules
examples/         → User examples
```

### Q: How do I test my changes?
```
pytest tests/ --cov=src tests/  # Coverage report
python src/train_baseline.py    # Full pipeline
python src/predict_example.py   # Interactive test
```

### Q: What coverage is required?
```
Minimum: 80% overall
Target: 85%+ for new code
Must fix: Any 0% modules
```

## 🎯 Good First Issues

Look for labels:
- `good-first-issue`
- `help-wanted`
- `documentation`
- `tests-needed`

**Perfect for beginners:**
1. Add docstrings to existing functions
2. Write tests for utils functions
3. Fix typos in documentation
4. Add example usage scripts

## 📚 Additional Resources

```
rules_files/QUICK_REFERENCE.md    - Daily checklists
rules_files/CODE_QUALITY_RULES.md - Full coding standards
rules_files/DATA_MODEL_RULES.md   - Data/model best practices
rules_files/PROJECT_RULES.md      - Project structure
```

## 🙌 Thank You!

Your contributions help make this the best sentiment analysis project on GitHub!

```
✅ Code improvements
✅ New datasets
✅ Better documentation
✅ More tests
✅ Performance optimizations
```

**Follow the rules → Your PR gets merged fast!** 🎉

---

**Last Updated**: December 2025  
**Project**: Multilingual Sentiment Analysis (1.4M samples)  
**Maintainer**: Hamza Siddiqui
```
