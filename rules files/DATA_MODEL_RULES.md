# Data & Model Rules for Sentiment Analysis Project

## Data Management Rules

### Data Pipeline Standards

#### Raw Data (data/raw/)
- ✅ **Read-only**: Never modify files in this directory
- ✅ **Versioning**: Include date in filename: `reviews_2025_12_24.csv`
- ✅ **Metadata**: Create accompanying `.json` with source, date, row count
- ✅ **Format**: Use standard formats (CSV, JSON, Parquet)
- ✅ **Encoding**: UTF-8 for text data
- ✅ **Size**: Document file size and creation date

#### Data Processing (data/processed/)
```python
# ✅ CORRECT - Document processing steps
import json
from datetime import datetime

def save_processed_data(df, name, processing_steps):
    """Save processed data with metadata."""
    
    filename = f"processed_{name}_{datetime.now().strftime('%Y_%m_%d')}.csv"
    filepath = Path('data/processed') / filename
    
    # Save data
    df.to_csv(filepath, index=False, encoding='utf-8')
    
    # Save metadata
    metadata = {
        'original_file': 'raw/reviews_2025_12_24.csv',
        'processing_date': datetime.now().isoformat(),
        'rows': len(df),
        'columns': list(df.columns),
        'processing_steps': processing_steps,
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict()
    }
    
    metadata_file = filepath.with_suffix('.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
```

### Data Validation Rules

#### Before Training
```python
# ✅ CORRECT - Comprehensive data validation
def validate_training_data(data: pd.DataFrame) -> bool:
    """Validate data before training."""
    
    # Check required columns
    required_cols = {'text', 'sentiment'}
    if not required_cols.issubset(data.columns):
        raise ValueError(
            f"Missing columns: {required_cols - set(data.columns)}"
        )
    
    # Check data types
    if not pd.api.types.is_string_dtype(data['text']):
        raise TypeError("'text' column must be string type")
    
    if not pd.api.types.is_string_dtype(data['sentiment']):
        raise TypeError("'sentiment' column must be string type")
    
    # Check for empty values
    if data['text'].isnull().any():
        print(f"Warning: {data['text'].isnull().sum()} empty texts")
    
    if data['sentiment'].isnull().any():
        print(f"Warning: {data['sentiment'].isnull().sum()} empty labels")
    
    # Check data distribution
    sentiment_dist = data['sentiment'].value_counts()
    print("Sentiment Distribution:")
    print(sentiment_dist)
    
    # Warn if severely imbalanced
    if sentiment_dist.max() / sentiment_dist.min() > 10:
        print("Warning: Severely imbalanced dataset detected")
    
    # Check text length
    text_lengths = data['text'].str.len()
    print(f"Text length stats:")
    print(f"  Min: {text_lengths.min()}")
    print(f"  Max: {text_lengths.max()}")
    print(f"  Mean: {text_lengths.mean():.1f}")
    
    return True
```

#### Text Quality Checks
```python
# ✅ CORRECT - Detect quality issues
def check_text_quality(data: pd.DataFrame) -> Dict[str, int]:
    """Identify potential text quality issues."""
    
    issues = {
        'empty_texts': (data['text'].str.len() == 0).sum(),
        'whitespace_only': (data['text'].str.strip().str.len() == 0).sum(),
        'too_short': (data['text'].str.len() < 3).sum(),
        'very_long': (data['text'].str.len() > 5000).sum(),
        'duplicates': data['text'].duplicated().sum(),
        'non_ascii': 0
    }
    
    # Check for non-ASCII
    for text in data['text']:
        try:
            text.encode('ascii')
        except UnicodeEncodeError:
            issues['non_ascii'] += 1
    
    return issues
```

### Data Preprocessing Standards

#### Cleaning Checklist
- ✅ **Normalization**: Convert to lowercase (or preserve case if needed)
- ✅ **Whitespace**: Remove extra spaces, tabs, newlines
- ✅ **URLs**: Remove or mark URLs consistently
- ✅ **Mentions**: Handle @mentions (remove or keep)
- ✅ **Hashtags**: Handle #hashtags consistently
- ✅ **Emojis**: Document handling strategy
- ✅ **Punctuation**: Document which to keep/remove
- ✅ **Numbers**: Document handling (keep/remove/replace)
- ✅ **Special Characters**: Remove or normalize
- ✅ **Duplicates**: Remove or track duplicates

#### Tokenization Standards
```python
# ✅ CORRECT - Document tokenization approach
def preprocess_text(
    text: str,
    remove_stopwords: bool = True,
    lowercase: bool = True
) -> str:
    """
    Preprocess text for sentiment analysis.
    
    Processing steps:
    1. Lowercase (optional)
    2. Remove URLs
    3. Remove mentions (@user)
    4. Remove hashtags (#tag)
    5. Handle contractions (don't -> do not)
    6. Remove special characters
    7. Remove extra whitespace
    8. Remove stopwords (optional)
    
    Parameters:
    -----------
    text : str
        Raw text to process
    remove_stopwords : bool
        Whether to remove common stopwords
    lowercase : bool
        Whether to convert to lowercase
        
    Returns:
    --------
    str
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Step 1: Lowercase
    if lowercase:
        text = text.lower()
    
    # Step 2-7: (implementation details)
    
    # Step 8: Remove stopwords
    if remove_stopwords:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        text = ' '.join(
            word for word in text.split() 
            if word not in stop_words
        )
    
    return text
```

---

## Model Training Rules

### Pre-Training Checklist

```python
# ✅ CORRECT - Comprehensive pre-training setup
def setup_training(config: Dict) -> Dict:
    """Configure training environment."""
    
    setup_info = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'environment': {
            'python_version': sys.version,
            'pandas_version': pd.__version__,
            'sklearn_version': sklearn.__version__,
        },
        'reproducibility': {
            'random_seed': config.get('random_seed', 42),
            'numpy_seed': config.get('random_seed', 42),
        }
    }
    
    # Set seeds for reproducibility
    np.random.seed(config['random_seed'])
    random.seed(config['random_seed'])
    
    return setup_info
```

### Train/Test Split Rules

#### Standard Split Strategy
```python
# ✅ CORRECT - Proper data splitting
def split_data(
    data: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify_by: str = 'sentiment'
) -> Dict[str, pd.DataFrame]:
    """
    Split data into train/validation/test sets.
    
    Uses stratified splitting to maintain class balance.
    
    Returns:
    --------
    Dict with 'train', 'val', 'test' DataFrames
    """
    
    # Validate proportions
    total = train_size + val_size + test_size
    assert abs(total - 1.0) < 0.01, "Proportions must sum to 1.0"
    
    # First split: train vs (val + test)
    X = data.drop(stratify_by, axis=1)
    y = data[stratify_by]
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        train_size=train_size,
        random_state=random_state,
        stratify=y
    )
    
    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=val_ratio,
        random_state=random_state,
        stratify=y_temp
    )
    
    return {
        'train': pd.concat([X_train, y_train], axis=1),
        'val': pd.concat([X_val, y_val], axis=1),
        'test': pd.concat([X_test, y_test], axis=1)
    }
```

### Model Versioning Rules

#### Naming Convention
```
models/
├── sentiment_model_v1.0_2025_12_24_acc_0.85.pkl
├── sentiment_model_v1.0_2025_12_24_metadata.json
├── sentiment_model_v1.1_2025_12_25_acc_0.87.pkl
└── sentiment_model_v1.1_2025_12_25_metadata.json

Naming pattern: {name}_v{version}_{date}_acc_{accuracy}.pkl
```

#### Model Metadata
```python
# ✅ CORRECT - Complete model metadata
def save_model_with_metadata(model, metrics, config, output_path):
    """Save model with complete metadata."""
    
    metadata = {
        'model_info': {
            'type': config['model_type'],
            'version': config['version'],
            'created': datetime.now().isoformat(),
            'author': config.get('author', 'Unknown'),
        },
        'training_config': config,
        'training_metrics': {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
        },
        'data_info': {
            'train_size': config['train_size'],
            'test_size': config['test_size'],
            'classes': config['classes'],
        },
        'environment': {
            'python_version': sys.version,
            'sklearn_version': sklearn.__version__,
            'numpy_version': np.__version__,
        },
        'hyperparameters': config.get('hyperparameters', {}),
        'preprocessing': config.get('preprocessing_steps', []),
    }
    
    # Save model
    with open(output_path / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
```

---

## Evaluation & Metrics Rules

### Metrics to Collect

```python
# ✅ CORRECT - Comprehensive evaluation
def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """Calculate all standard evaluation metrics."""
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(y_true, y_pred),
    }
    
    # Per-class metrics
    precision_per_class = precision_score(
        y_true, y_pred, average=None, labels=['positive', 'negative']
    )
    recall_per_class = recall_score(
        y_true, y_pred, average=None, labels=['positive', 'negative']
    )
    
    metrics['per_class_metrics'] = {
        'precision': {'positive': float(precision_per_class[0]),
                     'negative': float(precision_per_class[1])},
        'recall': {'positive': float(recall_per_class[0]),
                  'negative': float(recall_per_class[1])},
    }
    
    return metrics
```

### Logging Metrics
```python
# ✅ CORRECT - Save metrics with results
import json
from datetime import datetime

def save_experiment_results(metrics, config, output_dir):
    """Save complete experiment results."""
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'metrics': metrics,
    }
    
    filename = f"results_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.json"
    
    with open(output_dir / filename, 'w') as f:
        json.dump(results, f, indent=2)
```

---

**Document Version:** 1.0
**Last Updated:** December 2025
