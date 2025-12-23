# Code Style & Quality Rules for Sentiment Analysis Project

## Python Code Style Checklist

### Naming Conventions
```python
# ✅ CORRECT
class SentimentAnalyzer:
    """Analyzes sentiment of text data."""
    
    def __init__(self, model_type: str = 'logistic_regression'):
        self.model_type = model_type
        self.is_trained = False
        
    def predict_sentiment(self, text: str) -> str:
        """Predict sentiment for given text."""
        pass

RANDOM_SEED = 42
DEFAULT_TEST_SIZE = 0.2

# ❌ WRONG
class sentiment_analyzer:  # Should be PascalCase
    def __init__(self, MT='lr'):  # Unclear abbreviations
        self.mod = MT  # Bad variable name
        
def predictSentiment(t):  # Should be snake_case
    pass
```

### Imports Organization
```python
# ✅ CORRECT - Proper import order
import os
import sys
from pathlib import Path
from typing import List, Dict, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.sentiment_analyzer import SentimentAnalyzer
from src.utils.helpers import normalize_text

# ❌ WRONG - Mixed import order
from src.sentiment_analyzer import SentimentAnalyzer
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import sys
```

### Type Hints Requirements
```python
# ✅ CORRECT - Full type hints
def train_model(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Train a sentiment analysis model.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Training data with 'text' and 'sentiment' columns
    test_size : float
        Proportion of data for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing model metrics
    """
    pass

# ❌ WRONG - Missing type hints
def train_model(data, test_size=0.2):
    pass
```

### Docstring Format (Google Style)
```python
# ✅ CORRECT - Complete docstring
def predict(
    self,
    texts: Union[str, List[str]],
    return_confidence: bool = False
) -> Union[str, List[str]]:
    """
    Predict sentiment for text(s).
    
    This method handles both single strings and lists of strings,
    with optional confidence scores.
    
    Parameters:
    -----------
    texts : str or List[str]
        Single text or list of texts to analyze
    return_confidence : bool, optional
        If True, return confidence scores (default: False)
        
    Returns:
    --------
    str or List[str]
        Predicted sentiment label(s)
        
    Raises:
    -------
    ValueError
        If model has not been trained yet
    TypeError
        If texts is not str or List[str]
        
    Examples:
    ---------
    >>> analyzer = SentimentAnalyzer()
    >>> analyzer.train(data)
    >>> result = analyzer.predict("Great movie!")
    >>> print(result)
    'positive'
    """
    if not self.is_trained:
        raise ValueError("Model must be trained before predictions")
    pass

# ❌ WRONG - Incomplete docstring
def predict(self, texts, return_confidence=False):
    # Predict sentiment
    pass
```

### Line Length & Formatting
```python
# ✅ CORRECT - Under 88 characters
result = model.predict(
    text_data,
    return_confidence=True,
    normalize=False
)

# Better readability with long operations
data_processed = (
    df
    .dropna()
    .groupby('sentiment')
    .agg({'text': 'count'})
    .sort_values(ascending=False)
)

# ❌ WRONG - Line too long
result = model.predict(text_data, return_confidence=True, normalize=False, use_cache=True, batch_size=32)
```

### Error Handling
```python
# ✅ CORRECT - Specific exception handling
def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV data from file."""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Data file not found: {filepath}"
            )
        
        df = pd.read_csv(filepath)
        
        if df.empty:
            raise ValueError(
                f"Data file is empty: {filepath}"
            )
        
        return df
        
    except pd.errors.ParserError as e:
        raise ValueError(
            f"Error parsing CSV file {filepath}: {str(e)}"
        )

# ❌ WRONG - Too generic
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        return df
    except:  # Catches everything!
        pass
```

### Constants Definition
```python
# ✅ CORRECT - Constants at module level
MODULE_AUTHOR = "Your Name"
CURRENT_VERSION = "1.0.0"
MAX_TEXT_LENGTH = 512
RANDOM_SEED = 42
MODEL_SAVE_PATH = "models/"
BATCH_SIZE = 32

class TextPreprocessor:
    """Text preprocessing utility."""
    
    def __init__(self):
        self.max_length = MAX_TEXT_LENGTH
        self.seed = RANDOM_SEED

# ❌ WRONG - Magic numbers scattered in code
def preprocess(text):
    if len(text) > 512:  # Magic number!
        text = text[:512]
    return text
```

### Class Structure
```python
# ✅ CORRECT - Well-organized class
class SentimentAnalyzer:
    """
    Analyzes sentiment of text using machine learning models.
    
    Supports multiple algorithms and provides evaluation metrics.
    """
    
    def __init__(self, model_type: str = 'logistic_regression'):
        """Initialize analyzer with specified model type."""
        self.model_type = model_type
        self.model_pipeline = self._create_pipeline()
        self.is_trained = False
        self.evaluation_results = None
    
    # Public methods first
    def train(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train the sentiment model."""
        pass
    
    def predict(self, text: str) -> str:
        """Predict sentiment for text."""
        pass
    
    # Protected/private methods last
    def _create_pipeline(self):
        """Create ML pipeline (internal use)."""
        pass
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data format."""
        pass

# ❌ WRONG - Disorganized class
class SentimentAnalyzer:
    def __init__(self):
        pass
    
    def _internal_method(self):
        pass
    
    def train(self):
        pass
    
    def _another_internal(self):
        pass
```

### Function Length Rule
```python
# ✅ CORRECT - Small, focused functions
def calculate_metrics(
    y_true: List[str],
    y_pred: List[str]
) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# ❌ WRONG - Function too long with mixed concerns
def do_everything(data):  # 100+ lines mixing data cleaning, training, evaluation
    pass
```

---

## Code Quality Metrics

### Complexity Limits
- **Cyclomatic Complexity**: Maximum 10 per function
- **Cognitive Complexity**: Maximum 15 per function
- **Max Function Length**: 50 lines
- **Max Class Size**: 300 lines
- **Max Nesting Depth**: 3 levels

### Code Coverage Targets
- **Overall**: Minimum 80%
- **Critical Modules**: Minimum 90%
- **Utils**: Minimum 85%

### Performance Standards
- **Model Training**: Log progress every N iterations
- **Prediction**: Single prediction < 100ms
- **Batch Processing**: Efficient memory usage
- **Data Loading**: Handle datasets up to 1GB

---

## Common Mistakes to Avoid

### 1. Global Variables
```python
# ❌ BAD
model = None  # Global state

def load_model():
    global model
    model = pickle.load('model.pkl')

# ✅ GOOD
class ModelManager:
    def __init__(self):
        self.model = None
    
    def load(self):
        self.model = pickle.load('model.pkl')
```

### 2. Hardcoded Paths
```python
# ❌ BAD
df = pd.read_csv('/home/user/data/reviews.csv')

# ✅ GOOD
from pathlib import Path
DATA_DIR = Path(__file__).parent.parent / 'data'
df = pd.read_csv(DATA_DIR / 'reviews.csv')
```

### 3. Print Statements vs Logging
```python
# ❌ BAD
print("Model training started")
print("Error:", str(e))

# ✅ GOOD
import logging
logger = logging.getLogger(__name__)
logger.info("Model training started")
logger.error("Training error: %s", str(e))
```

### 4. String Concatenation
```python
# ❌ BAD
message = "Training " + model_type + " with " + str(epochs) + " epochs"

# ✅ GOOD
message = f"Training {model_type} with {epochs} epochs"
```

### 5. Mutable Default Arguments
```python
# ❌ BAD
def add_label(text, labels=[]):
    labels.append(text)
    return labels

# ✅ GOOD
def add_label(text, labels=None):
    if labels is None:
        labels = []
    labels.append(text)
    return labels
```

---

## Pre-Commit Checklist

Before committing code:
- [ ] All functions have type hints
- [ ] All functions have docstrings
- [ ] No hardcoded paths or credentials
- [ ] No print statements (use logging)
- [ ] No unused imports
- [ ] No comments explaining obvious code
- [ ] Functions under 50 lines
- [ ] Error handling for edge cases
- [ ] Tests pass
- [ ] Code follows naming conventions

---

**Document Version:** 1.0
**Last Updated:** December 2025
