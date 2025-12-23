# Testing & Deployment Rules for Sentiment Analysis Project

## Testing Requirements

### Unit Testing Structure

```python
# ✅ CORRECT - Proper test file organization
# File: tests/test_sentiment_analyzer.py

import pytest
import pandas as pd
from src.sentiment_analyzer import SentimentAnalyzer

class TestSentimentAnalyzer:
    """Test suite for SentimentAnalyzer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Provide test data."""
        return pd.DataFrame({
            'text': [
                "I love this movie!",
                "This is terrible.",
                "Amazing performance!",
                "Worst film ever."
            ],
            'sentiment': ['positive', 'negative', 'positive', 'negative']
        })
    
    @pytest.fixture
    def analyzer(self):
        """Provide initialized analyzer."""
        return SentimentAnalyzer(model_type='logistic_regression')
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.model_type == 'logistic_regression'
        assert analyzer.is_trained == False
        assert analyzer.model_pipeline is not None
    
    def test_train_valid_data(self, analyzer, sample_data):
        """Test training with valid data."""
        metrics = analyzer.train(sample_data)
        
        assert analyzer.is_trained == True
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_train_empty_data(self, analyzer):
        """Test training fails with empty data."""
        empty_data = pd.DataFrame({'text': [], 'sentiment': []})
        
        with pytest.raises(ValueError):
            analyzer.train(empty_data)
    
    def test_predict_single_text(self, analyzer, sample_data):
        """Test single prediction."""
        analyzer.train(sample_data)
        result = analyzer.predict("This is great!")
        
        assert isinstance(result, str)
        assert result in ['positive', 'negative']
    
    def test_predict_multiple_texts(self, analyzer, sample_data):
        """Test batch prediction."""
        analyzer.train(sample_data)
        texts = ["Great!", "Bad."]
        results = analyzer.predict(texts)
        
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(r in ['positive', 'negative'] for r in results)
    
    def test_predict_without_training(self):
        """Test prediction fails before training."""
        analyzer = SentimentAnalyzer()
        
        with pytest.raises(ValueError):
            analyzer.predict("Test text")
```

### Test Coverage Standards

#### Minimum Coverage by Module
```
src/sentiment_analyzer.py:       90%+
src/text_preprocessor.py:        85%+
src/train_models.py:             85%+
src/predict.py:                  85%+
src/utils/:                      80%+
```

#### Run Coverage Report
```bash
# Generate coverage report
pytest --cov=src --cov-report=html tests/

# View results
open htmlcov/index.html
```

### Edge Case Testing

```python
# ✅ CORRECT - Comprehensive edge case testing
class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_string(self, analyzer, sample_data):
        """Handle empty string input."""
        analyzer.train(sample_data)
        result = analyzer.predict("")
        assert isinstance(result, str)
    
    def test_none_input(self, analyzer, sample_data):
        """Handle None input gracefully."""
        analyzer.train(sample_data)
        with pytest.raises((TypeError, ValueError)):
            analyzer.predict(None)
    
    def test_very_long_text(self, analyzer, sample_data):
        """Handle very long texts."""
        analyzer.train(sample_data)
        long_text = "word " * 10000
        result = analyzer.predict(long_text)
        assert isinstance(result, str)
    
    def test_special_characters(self, analyzer, sample_data):
        """Handle special characters."""
        analyzer.train(sample_data)
        special_text = "Love it!!! 😍😍😍 @user #amazing"
        result = analyzer.predict(special_text)
        assert isinstance(result, str)
    
    def test_unicode_text(self, analyzer, sample_data):
        """Handle unicode characters."""
        analyzer.train(sample_data)
        unicode_text = "这是很好的电影 🎬"
        result = analyzer.predict(unicode_text)
        assert isinstance(result, str)
```

### Integration Testing

```python
# ✅ CORRECT - End-to-end pipeline testing
class TestIntegration:
    """Integration tests for complete pipelines."""
    
    def test_full_pipeline(self):
        """Test complete training and prediction pipeline."""
        # 1. Load and prepare data
        data = pd.DataFrame({
            'text': ["Great!", "Bad.", "Wonderful!", "Terrible."],
            'sentiment': ['positive', 'negative', 'positive', 'negative']
        })
        
        # 2. Train model
        analyzer = SentimentAnalyzer()
        metrics = analyzer.train(data)
        
        # 3. Evaluate
        assert metrics['accuracy'] > 0
        
        # 4. Make predictions
        test_texts = ["Amazing!", "Awful!"]
        results = analyzer.predict(test_texts)
        
        # 5. Verify results
        assert len(results) == len(test_texts)
        assert all(isinstance(r, str) for r in results)
    
    def test_model_persistence(self, tmp_path):
        """Test saving and loading models."""
        # Train and save
        analyzer = SentimentAnalyzer()
        analyzer.train(sample_data)
        model_path = tmp_path / "test_model.pkl"
        analyzer.save(model_path)
        
        # Load and verify
        loaded_analyzer = SentimentAnalyzer.load(model_path)
        assert loaded_analyzer.is_trained == True
        
        # Verify predictions are identical
        text = "Test text"
        original_pred = analyzer.predict(text)
        loaded_pred = loaded_analyzer.predict(text)
        assert original_pred == loaded_pred
```

---

## Testing Checklist

Before each commit:
- [ ] All tests pass: `pytest tests/`
- [ ] Coverage acceptable: `pytest --cov=src tests/`
- [ ] No print statements (use logging)
- [ ] Edge cases tested
- [ ] Error messages are clear
- [ ] Tests are independent (no shared state)
- [ ] Test names describe what they test
- [ ] Docstrings explain test purpose

---

## Deployment Rules

### Pre-Deployment Checklist

```
Code Quality:
- [ ] All tests pass
- [ ] Code coverage >= 80%
- [ ] Code style check: Black, Flake8
- [ ] Type checking: Mypy
- [ ] No security issues: Bandit
- [ ] No hardcoded credentials

Documentation:
- [ ] README.md updated
- [ ] API documentation complete
- [ ] Setup instructions clear
- [ ] Examples provided
- [ ] Troubleshooting section added

Model Quality:
- [ ] Model performance acceptable
- [ ] Cross-validation completed
- [ ] Metrics documented
- [ ] Hyperparameters documented
- [ ] Model size acceptable

Data Management:
- [ ] Data preprocessing documented
- [ ] Data validation in place
- [ ] Training/test split strategy clear
- [ ] No data leakage
- [ ] Data versioning setup

Infrastructure:
- [ ] Requirements.txt updated
- [ ] Dependencies pinned to versions
- [ ] Configuration files setup
- [ ] Logging configured
- [ ] Error handling complete
```

### Production Configuration

```python
# ✅ CORRECT - Production-ready configuration
# File: config/production.py

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_DIR = PROJECT_ROOT / 'models'
LOG_DIR = PROJECT_ROOT / 'logs'

# Model Settings
MODEL_TYPE = 'logistic_regression'
MODEL_VERSION = '1.0'
MODEL_PATH = MODEL_DIR / 'sentiment_model_v1.0.pkl'

# Data Settings
BATCH_SIZE = 32
MAX_TEXT_LENGTH = 512
TEXT_ENCODING = 'utf-8'

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Prediction Settings
CONFIDENCE_THRESHOLD = 0.7
TIMEOUT_SECONDS = 30

# Validation
MIN_ACCURACY = 0.80
MIN_F1_SCORE = 0.75

# Environment
ENVIRONMENT = os.getenv('ENV', 'production')
DEBUG = False
```

### Logging Setup

```python
# ✅ CORRECT - Production logging configuration
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from config.production import LOG_DIR, LOG_LEVEL, LOG_FORMAT

def setup_logging(name):
    """Setup production-grade logging."""
    
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # Create logs directory if needed
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        LOG_DIR / f'{name}.log',
        maxBytes=10_485_760,  # 10MB
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Usage
logger = setup_logging(__name__)
logger.info("Model loaded successfully")
```

### Version Control for Deployment

```bash
# Create deployment tag
git tag -a v1.0.0 -m "Production release v1.0.0"
git push origin v1.0.0

# Create release notes
# Document:
# - What changed
# - Improvements
# - Bug fixes
# - Breaking changes (if any)
# - Migration steps (if needed)
```

---

## Monitoring & Maintenance

### Post-Deployment Monitoring

#### Track These Metrics
- ✅ Model accuracy on live data
- ✅ Prediction latency
- ✅ Error rates
- ✅ User feedback accuracy
- ✅ Data drift detection

#### Logging in Production
```python
# ✅ CORRECT - Production logging
logger.info(
    "Prediction made",
    extra={
        'text_length': len(text),
        'prediction': prediction,
        'confidence': confidence,
        'response_time_ms': response_time,
        'user_id': user_id,
        'timestamp': datetime.now().isoformat()
    }
)
```

### Model Retraining Rules

#### When to Retrain
- ✅ Accuracy drops below MIN_ACCURACY threshold
- ✅ New significant data volume available
- ✅ Data distribution changes detected
- ✅ User feedback indicates poor performance
- ✅ Scheduled monthly/quarterly retraining

#### Retraining Checklist
```
- [ ] Gather new training data
- [ ] Validate data quality
- [ ] Run full test suite
- [ ] Compare with previous model
- [ ] If better: save new version
- [ ] Create deployment ticket
- [ ] Document changes
- [ ] Plan rollback strategy
```

---

## Performance Standards

### Model Performance Targets
```
Accuracy:     >= 80%
Precision:    >= 78%
Recall:       >= 75%
F1-Score:     >= 75%
Response Time: < 100ms per prediction
```

### Code Performance Standards
```
Test Suite:   Complete in < 60 seconds
Model Load:   < 2 seconds
Single Prediction: < 100ms
Batch (100 items): < 5 seconds
```

### System Requirements
```
Python:       3.8+
RAM:          2GB minimum
Disk:         1GB free for models/logs
CPU:          2 cores minimum
```

---

## Troubleshooting Guide

### Common Issues

#### Issue: Model Not Training
```python
# ✅ Check data first
if df.empty:
    raise ValueError("Dataset is empty")

if df[['text', 'sentiment']].isnull().any().any():
    print("Missing values found:")
    print(df.isnull().sum())

# Check data types
assert pd.api.types.is_string_dtype(df['text'])
assert pd.api.types.is_string_dtype(df['sentiment'])
```

#### Issue: Poor Accuracy
```python
# ✅ Investigate
1. Check class balance: df['sentiment'].value_counts()
2. Sample predictions: print(df['text'].sample(5))
3. Check preprocessing: verify text is cleaned consistently
4. Visualize confusion matrix
5. Try different hyperparameters
```

#### Issue: Slow Predictions
```python
# ✅ Optimize
1. Profile code: python -m cProfile -s cumtime script.py
2. Use batch predictions instead of single
3. Cache preprocessing results
4. Consider model quantization
5. Use GPU if available
```

---

**Document Version:** 1.0
**Last Updated:** December 2025
