# 📋 SENTIMENT ANALYSIS PROJECT - RULES INDEX

## Welcome! 👋

You now have **5 comprehensive rules documents** to keep your Sentiment Analysis project organized, maintainable, and professional.

---

## 📚 Your Rules Documents

### 1. **QUICK_REFERENCE.md** ⭐ START HERE
**Best for:** Quick lookups and checklists
- Quick start checklist
- Key rules at a glance
- Common mistakes to avoid
- Performance targets
- Deployment readiness checklist

👉 **When to use:** Before starting work each day

---

### 2. **PROJECT_RULES.md** 📁
**Best for:** Overall project structure and standards
- Project directory structure
- Code organization guidelines
- Naming conventions for files/folders
- Git & version control rules
- Performance & optimization standards

**Key Sections:**
- Project Structure Rules (how to organize files)
- Code Quality Rules (general standards)
- Data Handling Rules (data integrity)
- Model Training Rules (training best practices)
- Testing & Validation Rules (test coverage targets)
- Documentation Rules (what to document)
- Git & Version Control Rules (commit standards)
- Performance & Optimization Rules (speed targets)

👉 **When to use:** Setting up project, organizing code

---

### 3. **CODE_QUALITY_RULES.md** 🐍
**Best for:** Python code style and quality standards
- Naming conventions (PEP 8)
- Type hints requirements
- Docstring format (Google style)
- Line length & formatting
- Error handling patterns
- Class structure
- Common mistakes to avoid

**Key Code Examples:**
- ✅ CORRECT code examples
- ❌ WRONG code examples with explanations

**Metrics:**
- Cyclomatic complexity limits
- Code coverage targets
- Performance standards

👉 **When to use:** Writing Python code, doing code reviews

---

### 4. **DATA_MODEL_RULES.md** 🎯
**Best for:** Data handling and model management
- Data pipeline standards
  - Raw data handling (read-only)
  - Data processing (with metadata)
  - Data validation (before training)
- Text quality checks
- Data preprocessing standards
- Model training setup
- Train/test/validation split strategy
- Model versioning & naming
- Evaluation & metrics collection

**Key Code Examples:**
- Data validation functions
- Text preprocessing
- Metadata saving
- Model evaluation

👉 **When to use:** Loading data, training models, saving results

---

### 5. **TESTING_DEPLOYMENT_RULES.md** 🚀
**Best for:** Testing and deployment procedures
- Unit testing structure
- Test coverage standards by module
- Edge case testing
- Integration testing
- Pre-deployment checklist
- Production configuration
- Logging setup
- Monitoring & maintenance
- Model retraining rules
- Performance standards
- Troubleshooting guide

**Key Code Examples:**
- Pytest fixtures and tests
- Production logging configuration
- Edge case test examples
- Troubleshooting patterns

👉 **When to use:** Writing tests, deploying to production

---

## 🎯 How to Use These Rules

### Daily Development Workflow

#### 📖 Morning: Check Quick Reference
```bash
# Review what you need to do today
cat QUICK_REFERENCE.md
```
- Check the relevant checklist
- See performance targets
- Remember critical rules

#### 💻 Writing Code
```bash
# While writing Python
cat CODE_QUALITY_RULES.md
```
- Follow naming conventions
- Add type hints (copy examples if needed)
- Write docstrings (use Google style examples)

#### 🗂️ Organizing Files
```bash
# Setting up new modules/data
cat PROJECT_RULES.md
```
- Find correct directory structure
- Naming conventions for files
- Directory organization patterns

#### 📊 Handling Data
```bash
# Loading and processing data
cat DATA_MODEL_RULES.md
```
- Data validation checklist
- Preprocessing examples
- Model versioning format

#### ✅ Testing Code
```bash
# Writing and running tests
cat TESTING_DEPLOYMENT_RULES.md
```
- Unit test structure (copy the class template)
- Edge case examples
- Coverage targets

---

## 🔑 Critical Rules Summary

### 🚫 NEVER Do This
```
1. ❌ Modify files in data/raw/
2. ❌ Hardcode file paths
3. ❌ Use print() in production code
4. ❌ Train on test data
5. ❌ Commit model files to git
6. ❌ Skip input validation
7. ❌ Ignore error handling
8. ❌ Use global variables
9. ❌ Commit without tests passing
10. ❌ Deploy without documentation
```

### ✅ ALWAYS Do This
```
1. ✅ Set random seeds (reproducibility)
2. ✅ Write type hints (self-documenting)
3. ✅ Add docstrings (all functions)
4. ✅ Validate inputs (error prevention)
5. ✅ Use logging (debugging)
6. ✅ Save model metadata (traceability)
7. ✅ Version data/models (reproducibility)
8. ✅ Write tests (quality assurance)
9. ✅ Document assumptions (clarity)
10. ✅ Review code before merging (quality)
```

---

## 📊 File Organization Template

Use this structure for your project:
```
sentiment-analysis-project/
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── sentiment_analyzer.py
│   ├── text_preprocessor.py
│   ├── train_models.py
│   ├── predict.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
│
├── data/                         # Data files
│   ├── raw/                      # NEVER MODIFY
│   │   └── reviews_2025_12_24.csv
│   ├── processed/                # Your working data
│   │   └── processed_reviews_2025_12_24.csv
│   └── test/
│
├── models/                       # Trained models
│   ├── sentiment_model_v1.0_2025_12_24_acc_0.85.pkl
│   └── sentiment_model_v1.0_2025_12_24_metadata.json
│
├── tests/                        # Test files
│   ├── __init__.py
│   ├── test_sentiment_analyzer.py
│   ├── test_preprocessor.py
│   └── test_models.py
│
├── config/                       # Configuration
│   ├── __init__.py
│   ├── development.py
│   └── production.py
│
├── logs/                         # Application logs
│   └── app.log
│
├── results/                      # Experiment results
│   └── results_2025_12_24_15_30_45.json
│
├── examples/                     # Example scripts
│   └── basic_usage.py
│
├── notebooks/                    # Jupyter notebooks
│   └── exploration.ipynb
│
├── .gitignore
├── README.md
├── requirements.txt
├── QUICK_REFERENCE.md           ⭐ YOU ARE HERE
├── PROJECT_RULES.md
├── CODE_QUALITY_RULES.md
├── DATA_MODEL_RULES.md
└── TESTING_DEPLOYMENT_RULES.md
```

---

## 🚀 Quick Start Paths

### "I'm just starting"
1. Read: **QUICK_REFERENCE.md**
2. Read: **PROJECT_RULES.md** (Project Structure section)
3. Create project folders following the template above
4. Read: **CODE_QUALITY_RULES.md**

### "I'm writing code"
1. Check: **CODE_QUALITY_RULES.md** (naming, type hints, docstrings)
2. Reference: Examples in the document
3. Check: **QUICK_REFERENCE.md** (before committing)

### "I'm handling data"
1. Check: **DATA_MODEL_RULES.md**
2. Follow: Data validation checklist
3. Save: Metadata with processed data
4. Document: All preprocessing steps

### "I'm training a model"
1. Check: **DATA_MODEL_RULES.md** (Model Training section)
2. Follow: Pre-training checklist
3. Set: Random seeds
4. Save: Model with metadata
5. Log: All metrics

### "I'm writing tests"
1. Check: **TESTING_DEPLOYMENT_RULES.md**
2. Copy: Test class template
3. Write: Tests following pattern
4. Run: `pytest --cov=src tests/`
5. Verify: 80%+ coverage

### "I'm deploying"
1. Check: **TESTING_DEPLOYMENT_RULES.md** (Pre-Deployment section)
2. Run: Full checklist
3. Tag: Version in git
4. Deploy: Following checklist
5. Monitor: Following monitoring rules

---

## 📖 How to Reference Examples

All rules documents include code examples:

### ✅ CORRECT Examples
These show the right way to do things.
```python
# ✅ CORRECT
class SentimentAnalyzer:
    def __init__(self, model_type: str = 'logistic_regression'):
        self.model_type = model_type
        self.is_trained = False
```

**Copy this pattern when you write similar code.**

### ❌ WRONG Examples
These show what NOT to do.
```python
# ❌ WRONG
class sentiment_analyzer:  # Should be PascalCase
    def __init__(self, MT='lr'):  # Unclear abbreviations
        self.mod = MT  # Bad variable name
```

**Avoid this pattern.**

---

## 🎓 Learning Path

### Week 1: Foundations
- [ ] Read QUICK_REFERENCE.md completely
- [ ] Read PROJECT_RULES.md structure section
- [ ] Read CODE_QUALITY_RULES.md
- [ ] Create proper project structure
- [ ] Write first module following rules

### Week 2: Data & Models
- [ ] Read DATA_MODEL_RULES.md
- [ ] Load and validate data
- [ ] Train first model
- [ ] Save model with metadata
- [ ] Document all steps

### Week 3: Testing & Quality
- [ ] Read TESTING_DEPLOYMENT_RULES.md
- [ ] Write unit tests
- [ ] Achieve 80%+ coverage
- [ ] Run code quality checks
- [ ] Fix any issues

### Week 4: Production
- [ ] Set up logging
- [ ] Run pre-deployment checklist
- [ ] Tag version in git
- [ ] Deploy following procedures
- [ ] Monitor results

---

## 🤔 Common Questions

### Q: Where should I put my data?
**A:** See **PROJECT_RULES.md** - Project Structure section
- Raw data: `data/raw/` (read-only)
- Working data: `data/processed/`
- Test data: `data/test/`

### Q: How should I name variables?
**A:** See **CODE_QUALITY_RULES.md** - Naming Conventions
- Variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPERCASE`

### Q: Can I modify raw data files?
**A:** NO! See **DATA_MODEL_RULES.md** - Data Handling Rules
- Raw data is read-only
- Always work in `data/processed/`

### Q: What should model files be named?
**A:** See **DATA_MODEL_RULES.md** - Model Versioning Rules
- Pattern: `model_name_v1.0_2025_12_24_acc_0.85.pkl`

### Q: How much test coverage do I need?
**A:** See **TESTING_DEPLOYMENT_RULES.md** - Test Coverage Standards
- Minimum: 80%
- Critical modules: 90%
- Run: `pytest --cov=src tests/`

### Q: What's the deployment checklist?
**A:** See **TESTING_DEPLOYMENT_RULES.md** - Pre-Deployment Checklist
- Tests passing
- 80%+ coverage
- Code style clean
- Logging configured
- Etc.

---

## ✨ Pro Tips

1. **Keep this index file open** while working
2. **Bookmark the most relevant rule file** for your current task
3. **Copy code examples** from the documents
4. **Check checklists** before major changes
5. **Reference patterns** when writing similar code
6. **Ask "what does the rule say?"** when uncertain

---

## 📞 When to Reference

| Situation | Document |
|-----------|----------|
| "What should I do first?" | QUICK_REFERENCE.md |
| "How do I name this?" | CODE_QUALITY_RULES.md |
| "Where do I save files?" | PROJECT_RULES.md |
| "How do I preprocess data?" | DATA_MODEL_RULES.md |
| "How do I write tests?" | TESTING_DEPLOYMENT_RULES.md |
| "Can I modify raw data?" | DATA_MODEL_RULES.md |
| "How do I commit?" | PROJECT_RULES.md |
| "Is my code ready?" | QUICK_REFERENCE.md (Checklist) |
| "How do I deploy?" | TESTING_DEPLOYMENT_RULES.md |

---

## 🎉 You're Ready!

You now have everything you need to:

✅ Organize your project properly
✅ Write clean, professional code
✅ Handle data correctly
✅ Train and version models
✅ Write comprehensive tests
✅ Deploy with confidence
✅ Maintain code quality

**Next Step:** Pick the task you're starting with and reference the appropriate document!

---

**Created:** December 2025
**Version:** 1.0
**Status:** Ready to use

💡 **Pro Tip:** Bookmark this file and the QUICK_REFERENCE.md for daily use!
