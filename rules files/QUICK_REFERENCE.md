# QUICK REFERENCE - Sentiment Analysis Project Rules

## 📚 Rule Files Summary

Your project now has **4 comprehensive rules documents**:

1. **PROJECT_RULES.md** - Overall project structure and standards
2. **CODE_QUALITY_RULES.md** - Python code style and quality standards
3. **DATA_MODEL_RULES.md** - Data handling and model management
4. **TESTING_DEPLOYMENT_RULES.md** - Testing and deployment procedures

---

## 🚀 Quick Start Checklist

### Before Starting Development
- [ ] Create project structure (see PROJECT_RULES.md)
- [ ] Setup virtual environment
- [ ] Install dependencies with pinned versions
- [ ] Create `.gitignore` file
- [ ] Initialize git repository

### When Writing Code
- [ ] Follow naming conventions (PEP 8)
- [ ] Add type hints to all functions
- [ ] Write docstrings (Google style)
- [ ] Keep functions under 50 lines
- [ ] Use logging, not print statements
- [ ] Validate inputs at function entry
- [ ] Handle edge cases

### When Handling Data
- [ ] NEVER modify raw data files
- [ ] Always work in `data/processed/` directory
- [ ] Save metadata with processed data
- [ ] Validate data before training
- [ ] Check for data leakage
- [ ] Document all preprocessing steps
- [ ] Version your datasets

### When Training Models
- [ ] Set random seeds for reproducibility
- [ ] Use stratified train/test split
- [ ] Save models with metadata
- [ ] Log all metrics and hyperparameters
- [ ] Document why choices were made
- [ ] Version your models with accuracy scores
- [ ] Never train on test data

### When Testing Code
- [ ] Write unit tests for all functions
- [ ] Test edge cases and error conditions
- [ ] Aim for 80%+ code coverage
- [ ] Use pytest framework
- [ ] Test complete pipelines (integration)
- [ ] Document test purpose in docstrings

### Before Committing
- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] No hardcoded paths or credentials
- [ ] No unused imports or variables
- [ ] Meaningful commit message
- [ ] Related documentation updated

### Before Deployment
- [ ] All tests pass with 80%+ coverage
- [ ] Model meets performance targets
- [ ] Requirements.txt has pinned versions
- [ ] Configuration documented
- [ ] Logging setup complete
- [ ] Error handling in place
- [ ] README is up-to-date

---

## 📋 Key Rules at a Glance

### Project Structure
```
✅ src/            - Source code only
✅ data/raw/       - Never modify, read-only
✅ data/processed/ - Your working data
✅ models/         - Save trained models here
✅ tests/          - Unit and integration tests
✅ config/         - Configuration files
✅ logs/           - Log files
✅ results/        - Metrics and results
```

### Code Standards
```
✅ Use snake_case for variables: my_variable
✅ Use PascalCase for classes: MyClass
✅ Use UPPERCASE for constants: MY_CONSTANT
✅ Max line length: 88 characters
✅ Type hints: def func(x: int) -> str:
✅ Docstrings: Google style for all public functions
✅ No magic numbers: Use constants instead
✅ No global variables: Use classes
✅ No print(): Use logging module
✅ No hardcoded paths: Use Path from pathlib
```

### Data Rules
```
✅ NEVER modify data/raw/ files
✅ Always validate data before using
✅ Check for missing values
✅ Document preprocessing steps
✅ Check for duplicate rows
✅ Handle class imbalance
✅ Save processed data with metadata
✅ Version datasets with dates
✅ Check for data leakage
```

### Model Rules
```
✅ Set seeds: np.random.seed(42)
✅ Use stratified split for classification
✅ Never train on test data
✅ Save models with full metadata
✅ Version models: name_v1.0_acc_0.85.pkl
✅ Document hyperparameters
✅ Track experiments
✅ Log metrics
✅ Compare models systematically
```

### Testing Rules
```
✅ Write tests for all functions
✅ Use pytest framework
✅ Test edge cases (empty, None, etc.)
✅ Aim for 80%+ coverage
✅ Test complete pipelines
✅ Use pytest.fixture for setup
✅ Write clear test names
✅ Test error conditions
```

---

## 🔍 Common Mistakes to Avoid

❌ **DON'T** modify raw data files
❌ **DON'T** hardcode file paths
❌ **DON'T** use print() in production code
❌ **DON'T** use global variables
❌ **DON'T** train on test data
❌ **DON'T** skip input validation
❌ **DON'T** use magic numbers
❌ **DON'T** mix concerns in one function
❌ **DON'T** ignore error handling
❌ **DON'T** commit model files to git

---

## 📊 Performance Targets

### Model Accuracy
- Minimum: 80%
- Target: 85%+

### Code Quality
- Test Coverage: 80%+
- Code Review: Required before merge
- Static Analysis: Pass Flake8, Mypy

### Execution Speed
- Model Load: < 2 seconds
- Single Prediction: < 100ms
- Batch Prediction (100 items): < 5 seconds
- Test Suite: < 60 seconds

---

## 🔄 Development Workflow

### 1. Create Feature Branch
```bash
git checkout -b feature/new-feature
```

### 2. Write Code Following Rules
- Follow naming conventions
- Add type hints and docstrings
- Use constants for magic numbers
- Validate inputs

### 3. Write Tests
- Unit tests for new functions
- Integration tests for pipelines
- Edge case tests

### 4. Run Quality Checks
```bash
# Tests
pytest tests/

# Coverage
pytest --cov=src tests/

# Style
black src/
flake8 src/

# Type checking
mypy src/
```

### 5. Commit with Clear Message
```bash
git add .
git commit -m "[Feature] Add new sentiment analyzer"
```

### 6. Push and Create Pull Request
```bash
git push origin feature/new-feature
```

### 7. Code Review
- Request review from teammate
- Address feedback
- Ensure all tests pass

### 8. Merge to Main
```bash
# Merge on GitHub after review
# Delete feature branch
git branch -d feature/new-feature
```

---

## 📝 File Naming Conventions

### Python Files
```
✅ sentiment_analyzer.py       - Classes and main logic
✅ text_preprocessor.py        - Preprocessing utilities
✅ train_models.py             - Training logic
✅ test_sentiment_analyzer.py  - Tests (prefix with 'test_')
```

### Data Files
```
✅ reviews_2025_12_24.csv           - Raw data with date
✅ processed_reviews_2025_12_24.csv - Processed data
✅ processed_reviews_2025_12_24.json - Metadata file
```

### Model Files
```
✅ sentiment_model_v1.0_2025_12_24_acc_0.85.pkl  - Model file
✅ sentiment_model_v1.0_2025_12_24_metadata.json - Metadata
```

### Results Files
```
✅ results_2025_12_24_15_30_45.json - Experiment results with timestamp
```

---

## 🛠️ Tools & Libraries to Use

### Core Data Science
- pandas - Data manipulation
- numpy - Numerical computing
- scikit-learn - Machine learning

### Testing
- pytest - Unit testing framework
- pytest-cov - Coverage reporting

### Code Quality
- black - Code formatter
- flake8 - Style checker
- mypy - Type checker
- pylint - Code analysis

### Logging
- logging (built-in) - Application logging

### Configuration
- json - Configuration files
- yaml - Alternative config format

---

## 📖 Documentation Requirements

### README.md Must Include
- [ ] Project description and goals
- [ ] Installation instructions
- [ ] Quick start example
- [ ] Data format requirements
- [ ] Model performance metrics
- [ ] Usage examples
- [ ] Troubleshooting section
- [ ] Contributing guidelines

### Code Documentation
- [ ] All functions have docstrings
- [ ] All parameters documented
- [ ] Return values documented
- [ ] Examples provided for public methods
- [ ] Error handling documented

### Results Documentation
- [ ] Model metrics saved
- [ ] Hyperparameters recorded
- [ ] Training conditions noted
- [ ] Data version specified
- [ ] Preprocessing steps listed

---

## 🚨 Critical Rules (MUST FOLLOW)

1. **NEVER modify data/raw/ directory**
   - This is your golden copy
   - Create copies in data/processed/ only

2. **ALWAYS set random seeds**
   - Ensures reproducibility
   - `np.random.seed(42)`, `random.seed(42)`

3. **NEVER train on test data**
   - Causes overfitting
   - Metrics become meaningless

4. **ALWAYS validate data first**
   - Check for nulls, duplicates, imbalance
   - Document findings

5. **ALWAYS save model metadata**
   - Include accuracy, date, config
   - Makes models traceable

6. **NEVER hardcode credentials**
   - Use environment variables
   - Keep .env in .gitignore

7. **ALWAYS write type hints**
   - Functions are self-documenting
   - Helps catch bugs early

8. **NEVER commit large files**
   - Use git-lfs or DVC for data/models
   - Keep repository lean

---

## 📞 Quick Reference Links

- **Python Style**: PEP 8 (https://pep8.org/)
- **Google Docstring**: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
- **pytest Docs**: https://docs.pytest.org/
- **scikit-learn**: https://scikit-learn.org/
- **pandas**: https://pandas.pydata.org/

---

## 🎯 Deployment Readiness Checklist

- [ ] All tests pass: `pytest tests/`
- [ ] Coverage >= 80%: `pytest --cov=src tests/`
- [ ] Code formatted: `black src/`
- [ ] No style issues: `flake8 src/`
- [ ] Types correct: `mypy src/`
- [ ] No hardcoded credentials
- [ ] Model meets accuracy target
- [ ] Logging configured
- [ ] Error handling complete
- [ ] Documentation up-to-date
- [ ] README includes setup steps
- [ ] Requirements.txt has versions
- [ ] Config files documented
- [ ] Model versioning clear

---

## 📅 When to Reference Rules

**At the START of development:**
→ Read PROJECT_RULES.md for structure

**When WRITING CODE:**
→ Reference CODE_QUALITY_RULES.md

**When HANDLING DATA:**
→ Check DATA_MODEL_RULES.md

**When TESTING/DEPLOYING:**
→ Use TESTING_DEPLOYMENT_RULES.md

**When STUCK:**
→ Check the troubleshooting section

---

**Version:** 1.0
**Created:** December 2025
**Keep this file in your project root for easy reference**
