# Sentiment Analysis Project - Rules & Guidelines

## 📋 Table of Contents
1. [Project Structure Rules](#project-structure-rules)
2. [Code Quality Rules](#code-quality-rules)
3. [Data Handling Rules](#data-handling-rules)
4. [Model Training Rules](#model-training-rules)
5. [Testing & Validation Rules](#testing--validation-rules)
6. [Documentation Rules](#documentation-rules)
7. [Git & Version Control Rules](#git--version-control-rules)
8. [Performance & Optimization Rules](#performance--optimization-rules)

---

## Project Structure Rules

### Directory Organization
```
sentiment-analysis-project/
├── src/                    # Source code
│   ├── sentiment_analyzer.py      # Main analyzer class
│   ├── text_preprocessor.py       # Text cleaning & preprocessing
│   ├── train_models.py            # Model training logic
│   ├── predict.py                 # Prediction module
│   └── utils/                     # Utility functions
├── data/                   # Data files
│   ├── raw/               # Original unmodified data
│   ├── processed/         # Cleaned data
│   └── test/              # Test datasets
├── models/                # Trained models storage
├── notebooks/             # Jupyter notebooks for exploration
├── tests/                 # Unit tests
├── examples/              # Example usage scripts
├── config/                # Configuration files
├── logs/                  # Log files
├── results/               # Output results & metrics
└── README.md              # Project documentation
```

### Rules
- ✅ Keep source code in `src/` directory only
- ✅ Store all data in `data/` directory, never in root or src
- ✅ Keep trained models in `models/` directory with clear naming
- ✅ All raw data goes to `data/raw/` - NEVER modify original files
- ✅ Use descriptive folder names in lowercase with underscores
- ✅ Create `__init__.py` in all Python packages
- ✅ Keep one logical module per file

---

## Code Quality Rules

### Python Standards
- ✅ Follow PEP 8 naming conventions
  - Variables & functions: `lowercase_with_underscores`
  - Classes: `PascalCase`
  - Constants: `UPPERCASE_WITH_UNDERSCORES`
- ✅ Maximum line length: 88 characters (Black formatter standard)
- ✅ Use type hints for all function parameters and returns
- ✅ Write docstrings for all classes and functions (Google style)
- ✅ Import order: Standard library → Third-party → Local modules
- ✅ No unused imports or variables
- ✅ Use meaningful variable names (no single letters except loops)

### Documentation
- ✅ Every function must have a docstring with:
  - Brief description
  - Parameters with types
  - Return value with type
  - Example usage (for public methods)
- ✅ Class docstrings must explain purpose and key methods
- ✅ Add comments for complex logic (WHY not WHAT)
- ✅ No commented-out code blocks - delete if not needed

### Code Organization
- ✅ Keep functions under 50 lines (split if longer)
- ✅ Keep classes focused on single responsibility
- ✅ Use private methods (underscore prefix) for internal logic
- ✅ Use constants at module level, not magic numbers
- ✅ Avoid deeply nested conditions (max 3 levels)

### Error Handling
- ✅ Raise specific exceptions, not generic ones
- ✅ Provide meaningful error messages
- ✅ Validate inputs at function entry
- ✅ Handle edge cases (empty data, None values, etc.)
- ✅ Log errors with context information
- ✅ Use try-except blocks wisely, not for flow control

---

## Data Handling Rules

### Data Integrity
- ✅ NEVER modify raw data files - create copies in `data/processed/`
- ✅ Track data source and date in metadata
- ✅ Maintain data versioning (data_v1, data_v2, etc.)
- ✅ Document all preprocessing steps
- ✅ Keep original data accessible for reproducibility

### Data Processing
- ✅ Validate data before processing (check types, ranges, nulls)
- ✅ Handle missing values explicitly (document strategy)
- ✅ Check for duplicate rows and handle appropriately
- ✅ Document data distribution before/after preprocessing
- ✅ Use consistent encoding (UTF-8 preferred)
- ✅ Preserve original text in separate column if cleaning
- ✅ Check data leakage between train/test sets

### DataFrame Requirements
- ✅ Use consistent column naming (lowercase with underscores)
- ✅ Include data type specification when loading
- ✅ Check for outliers and document treatment
- ✅ Balance datasets if needed (especially for classification)
- ✅ Include metadata about data shape and quality

### Text Data Specific
- ✅ Store raw text in separate column before cleaning
- ✅ Document all text preprocessing steps
- ✅ Track vocabulary size and changes
- ✅ Preserve case information if relevant
- ✅ Handle special characters, emojis, URLs consistently

---

## Model Training Rules

### Before Training
- ✅ Define clear objectives (accuracy targets, F1-score, etc.)
- ✅ Plan train/test split strategy before starting
- ✅ Document baseline metrics
- ✅ Set random seeds for reproducibility
  ```python
  np.random.seed(42)
  random.seed(42)
  tf.random.set_seed(42) # if using TensorFlow
  ```
- ✅ Version control all hyperparameters

### During Training
- ✅ Use stratified split for imbalanced datasets
- ✅ Log training progress and metrics
- ✅ Use consistent train/test/validation split (typically 70/15/15)
- ✅ Never train on test data
- ✅ Monitor for overfitting
- ✅ Save model checkpoints during training
- ✅ Document all model configurations

### Model Management
- ✅ Save trained models with metadata:
  - Model type and parameters
  - Training date
  - Training accuracy/metrics
  - Data version used
- ✅ Use consistent naming: `model_name_date_accuracy.pkl`
- ✅ Store models in `models/` with version numbers
- ✅ Keep model artifacts with code version tag
- ✅ Document which model version is production-ready

### Hyperparameter Tracking
- ✅ Create config file for all hyperparameters
- ✅ Document why each parameter was chosen
- ✅ Compare results across different hyperparameter sets
- ✅ Use grid search or random search systematically
- ✅ Record all hyperparameter experiments

---

## Testing & Validation Rules

### Unit Testing
- ✅ Write tests for all utility functions
- ✅ Test edge cases (empty data, None, negative numbers)
- ✅ Aim for at least 80% code coverage
- ✅ Use pytest framework
- ✅ Name test files: `test_module_name.py`
- ✅ Name test functions: `test_what_it_does()`
- ✅ Use descriptive assertion messages

### Integration Testing
- ✅ Test complete pipelines end-to-end
- ✅ Test data flow between modules
- ✅ Verify output shapes and types
- ✅ Test with different data sizes

### Model Validation
- ✅ Always use separate test set (never seen during training)
- ✅ Calculate multiple metrics: accuracy, precision, recall, F1
- ✅ Generate confusion matrix for classification
- ✅ Cross-validate with k-fold (k=5 recommended)
- ✅ Document performance on different data subsets
- ✅ Test on representative edge cases

### Validation Checklist
- ✅ Predictions make logical sense
- ✅ Model handles empty inputs gracefully
- ✅ Model handles unusual inputs without crashing
- ✅ Output format is consistent
- ✅ Performance is reproducible

---

## Documentation Rules

### README.md Requirements
- ✅ Project overview and goals
- ✅ Installation instructions
- ✅ Usage examples
- ✅ Data requirements and format
- ✅ Model performance metrics
- ✅ Troubleshooting section
- ✅ Contributing guidelines

### Code Comments
- ✅ Document WHY not WHAT
- ✅ Keep comments updated with code
- ✅ Use clear, professional language
- ✅ Avoid over-commenting obvious code

### Configuration Documentation
- ✅ Document all config parameters
- ✅ Provide default values with justification
- ✅ Explain impact of changing each parameter
- ✅ Keep config examples in README

### Results & Metrics
- ✅ Save metrics in results/ folder with date
- ✅ Include timestamp for all experiments
- ✅ Document dataset version used
- ✅ Include hyperparameters in results file
- ✅ Keep comparison table of model performance

---

## Git & Version Control Rules

### Commit Standards
- ✅ Write clear, descriptive commit messages
- ✅ Use present tense: "Add feature" not "Added feature"
- ✅ Keep commits focused on single change
- ✅ Format: `[Type] Brief description`
  - Types: `[Feature]`, `[Fix]`, `[Docs]`, `[Refactor]`, `[Test]`

### Branching Strategy
- ✅ Use feature branches: `feature/feature-name`
- ✅ Use bugfix branches: `bugfix/bug-name`
- ✅ Keep main/master branch stable and tested
- ✅ Require code review before merging to main
- ✅ Delete branches after merging

### .gitignore Rules
- ✅ Ignore data files: `data/raw/*`, `data/processed/*`
- ✅ Ignore model files: `models/*.pkl`, `models/*.h5`
- ✅ Ignore logs: `logs/*.log`
- ✅ Ignore virtual environment: `venv/`, `.venv/`
- ✅ Ignore cache: `__pycache__/`, `.pytest_cache/`
- ✅ Ignore IDE: `.vscode/`, `.idea/`
- ✅ Ignore OS files: `.DS_Store`, `Thumbs.db`

### What NOT to Commit
- ❌ Raw data files (use DVC or LFS instead)
- ❌ Trained model files (too large)
- ❌ Virtual environments
- ❌ API keys or credentials
- ❌ System/IDE files
- ❌ Large binary files

---

## Performance & Optimization Rules

### Code Performance
- ✅ Use vectorized operations (NumPy/Pandas) instead of loops
- ✅ Profile code for bottlenecks before optimizing
- ✅ Cache expensive computations
- ✅ Use generators for large datasets
- ✅ Minimize copying of large data structures

### Memory Management
- ✅ Load data in chunks if memory-constrained
- ✅ Release unused objects/connections
- ✅ Monitor memory usage during training
- ✅ Use sparse matrices for high-dimensional data

### Model Performance
- ✅ Document inference time requirements
- ✅ Test model on different hardware
- ✅ Optimize model size if needed
- ✅ Consider quantization for mobile deployment
- ✅ Benchmark against baseline models

### Logging & Monitoring
- ✅ Use logging module (not print statements)
- ✅ Set appropriate log levels: DEBUG, INFO, WARNING, ERROR
- ✅ Log important events and metrics
- ✅ Include timestamps in logs
- ✅ Rotate log files to prevent disk bloat

### Reproducibility
- ✅ Document Python and library versions
- ✅ Use requirements.txt with pinned versions
- ✅ Set random seeds at project start
- ✅ Document hardware used for training
- ✅ Provide exact preprocessing steps

---

## Checklist Before Deployment

- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] Documentation is complete and up-to-date
- [ ] Model performance meets targets
- [ ] Edge cases are handled
- [ ] Logging is in place
- [ ] Requirements.txt is updated
- [ ] No hardcoded paths or credentials
- [ ] Data pipeline is documented
- [ ] Model versioning is clear
- [ ] Performance is acceptable
- [ ] Code review completed

---

## Quick Reference: Do's and Don'ts

### DO ✅
- Use constants for magic numbers
- Write docstrings for all functions
- Validate inputs at function entry
- Use meaningful variable names
- Keep functions small and focused
- Document your assumptions
- Test edge cases
- Version your models
- Use type hints

### DON'T ❌
- Modify raw data files
- Hardcode file paths
- Use global variables
- Write overly complex functions
- Skip error handling
- Mix concerns in one function
- Use non-descriptive names
- Train on test data
- Ignore data leakage
- Deploy without testing

---

**Last Updated:** December 2025
**Version:** 1.0
