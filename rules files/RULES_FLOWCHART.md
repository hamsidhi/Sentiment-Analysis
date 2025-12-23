# 🎯 RULES USAGE FLOWCHART & GUIDE

## When You're Starting a Task - Which Document to Open?

```
START HERE
    |
    ├─→ "I just got this project"
    |       └─→ READ: README_RULES.md (complete overview)
    |           Then: PROJECT_RULES.md (Project Structure)
    |
    ├─→ "I need to write Python code"
    |       └─→ CHECK: CODE_QUALITY_RULES.md
    |           COPY: Naming conventions & docstring examples
    |           VERIFY: Type hints, error handling
    |
    ├─→ "I need to load/process data"
    |       └─→ CHECK: DATA_MODEL_RULES.md
    |           FOLLOW: Data validation checklist
    |           REMEMBER: raw/ is read-only!
    |
    ├─→ "I need to train a model"
    |       └─→ CHECK: DATA_MODEL_RULES.md
    |           FOLLOW: Model training setup
    |           SAVE: Model with metadata
    |
    ├─→ "I need to write tests"
    |       └─→ CHECK: TESTING_DEPLOYMENT_RULES.md
    |           COPY: Test class template
    |           TARGET: 80%+ coverage
    |
    ├─→ "I'm ready to commit code"
    |       └─→ CHECK: QUICK_REFERENCE.md
    |           RUN: Pre-commit checklist
    |
    └─→ "I'm ready to deploy"
            └─→ CHECK: TESTING_DEPLOYMENT_RULES.md
                RUN: Pre-deployment checklist
```

---

## Document Quick Links

### 🔴 CRITICAL ISSUES? Start Here:

| Issue | Document | Section |
|-------|----------|---------|
| "Is this allowed?" | PROJECT_RULES.md | "DON'T" section |
| "How do I name this?" | CODE_QUALITY_RULES.md | Naming Conventions |
| "Can I modify raw data?" | DATA_MODEL_RULES.md | Data Integrity |
| "How do I save models?" | DATA_MODEL_RULES.md | Model Versioning |
| "What tests do I write?" | TESTING_DEPLOYMENT_RULES.md | Unit Testing |
| "Is my code ready?" | TESTING_DEPLOYMENT_RULES.md | Pre-Deployment |

---

## Daily Usage Pattern

### 🌅 Morning (5 minutes)
```
1. Open: QUICK_REFERENCE.md
2. Check: Quick Start Checklist (your task)
3. Remember: Critical Rules section
4. Open: Specific rule document for your task
```

### 💻 Coding (ongoing)
```
1. Before writing: Check naming conventions
2. While writing: Copy docstring template
3. Before testing: Reference error handling examples
4. Before commit: Run pre-commit checklist
```

### 🌙 Evening (before committing)
```
1. Run tests: pytest tests/
2. Check coverage: pytest --cov=src tests/
3. Verify: QUICK_REFERENCE.md Pre-Commit Checklist
4. Commit with meaningful message
```

---

## Code Writing Workflow

### Step 1: Plan (2 min)
```
Check: PROJECT_RULES.md
Where should this file go?
  src/              → Main code
  tests/            → Test code
  utils/            → Helper functions
  config/           → Configuration
```

### Step 2: Start (5 min)
```
Check: CODE_QUALITY_RULES.md
Copy template:
  - Naming convention
  - Type hints
  - Docstring (Google style)
  - Error handling pattern
```

### Step 3: Write (30+ min)
```
Follow: CODE_QUALITY_RULES.md
- Use snake_case for variables
- PascalCase for classes
- UPPERCASE for constants
- Keep functions < 50 lines
- Validate inputs
- Handle errors
- Use logging (not print)
```

### Step 4: Test (10+ min)
```
Check: TESTING_DEPLOYMENT_RULES.md
- Write unit tests
- Test edge cases
- Test error conditions
- Run: pytest --cov=src tests/
- Target: 80%+ coverage
```

### Step 5: Review (5 min)
```
Check: QUICK_REFERENCE.md
- Code follows naming conventions ✓
- Functions have type hints ✓
- Docstrings complete ✓
- Tests pass ✓
- No hardcoded paths ✓
- No print statements ✓
- Clear variable names ✓
```

### Step 6: Commit (2 min)
```
git add .
git commit -m "[Feature] Description of changes"
```

---

## Common Task Workflows

### WORKFLOW 1: Load & Process Data

```
1. Check: DATA_MODEL_RULES.md
   - Data pipeline standards
   - Validation rules
   - Preprocessing examples

2. Steps:
   a) Load from data/raw/
   b) Validate (check columns, types, nulls)
   c) Process (clean text, handle missing)
   d) Save to data/processed/ with metadata
   e) Log statistics

3. Save with metadata:
   {
     "source": "data/raw/reviews_2025_12_24.csv",
     "processing_steps": [
       "Removed 5 null values",
       "Cleaned special characters",
       "Balanced classes (50/50)"
     ],
     "rows": 1000,
     "columns": ["text", "sentiment"]
   }

4. Verify: Use examples from DATA_MODEL_RULES.md
```

### WORKFLOW 2: Train a Model

```
1. Check: DATA_MODEL_RULES.md (Model Training)
   - Setup checklist
   - Train/test split
   - Hyperparameter tracking

2. Steps:
   a) Set random seeds (reproducibility!)
   b) Load processed data
   c) Split: 70% train, 15% val, 15% test
   d) Train model
   e) Evaluate
   f) Save with metadata

3. Save model with metadata:
   {
     "model_type": "logistic_regression",
     "version": "1.0",
     "date": "2025-12-24",
     "accuracy": 0.85,
     "precision": 0.83,
     "recall": 0.82,
     "f1_score": 0.82,
     "hyperparameters": {...}
   }

4. File name: sentiment_model_v1.0_2025_12_24_acc_0.85.pkl

5. Verify: All metrics logged and documented
```

### WORKFLOW 3: Write Tests

```
1. Check: TESTING_DEPLOYMENT_RULES.md
   - Test structure (pytest)
   - Fixtures
   - Edge case examples

2. Create: tests/test_your_module.py

3. Copy template:
   import pytest
   
   class TestYourClass:
       @pytest.fixture
       def sample_data(self):
           return {...}
       
       def test_specific_behavior(self, sample_data):
           # Arrange, Act, Assert
           result = function(sample_data)
           assert result == expected

4. Write tests for:
   ✓ Normal cases
   ✓ Edge cases (empty, None, extreme)
   ✓ Error conditions (should raise)
   ✓ Different input types

5. Run: pytest --cov=src tests/
   Target: 80%+ coverage

6. If coverage < 80%:
   - Add more edge case tests
   - Test error conditions
   - Test all branches
```

### WORKFLOW 4: Deploy

```
1. Check: TESTING_DEPLOYMENT_RULES.md
   - Pre-deployment checklist
   - Production config
   - Logging setup

2. Pre-deployment (30 min):
   ✓ All tests pass
   ✓ 80%+ coverage
   ✓ Code formatted (black)
   ✓ No style issues (flake8)
   ✓ Types correct (mypy)
   ✓ Model meets accuracy target
   ✓ No hardcoded credentials
   ✓ Logging configured
   ✓ Error handling complete
   ✓ README updated
   ✓ Requirements.txt has versions

3. Tag version:
   git tag -a v1.0.0 -m "Production release"
   git push origin v1.0.0

4. Deploy following your process

5. Monitor:
   - Check logs for errors
   - Monitor prediction accuracy
   - Track user feedback
```

---

## Decision Tree: Which Rule Document?

```
Do you have a question?
│
├─ "Is this allowed/forbidden?"
│  └─→ PROJECT_RULES.md (Rules section)
│
├─ "How do I organize this?"
│  └─→ PROJECT_RULES.md (Project Structure)
│
├─ "How do I name this?"
│  └─→ CODE_QUALITY_RULES.md (Naming Conventions)
│
├─ "How do I write this?"
│  └─→ CODE_QUALITY_RULES.md (Examples)
│
├─ "What's the format?"
│  ├─ Type hints? → CODE_QUALITY_RULES.md
│  ├─ Docstrings? → CODE_QUALITY_RULES.md
│  ├─ Commit? → PROJECT_RULES.md
│  └─ Model files? → DATA_MODEL_RULES.md
│
├─ "How do I handle data?"
│  ├─ Load? → DATA_MODEL_RULES.md
│  ├─ Validate? → DATA_MODEL_RULES.md
│  ├─ Save? → DATA_MODEL_RULES.md
│  └─ Version? → DATA_MODEL_RULES.md
│
├─ "How do I train?"
│  ├─ Setup? → DATA_MODEL_RULES.md
│  ├─ Split? → DATA_MODEL_RULES.md
│  ├─ Save? → DATA_MODEL_RULES.md
│  └─ Version? → DATA_MODEL_RULES.md
│
├─ "How do I test?"
│  ├─ Write tests? → TESTING_DEPLOYMENT_RULES.md
│  ├─ Edge cases? → TESTING_DEPLOYMENT_RULES.md
│  ├─ Coverage? → TESTING_DEPLOYMENT_RULES.md
│  └─ Run tests? → TESTING_DEPLOYMENT_RULES.md
│
└─ "How do I deploy?"
   ├─ Checklist? → TESTING_DEPLOYMENT_RULES.md
   ├─ Config? → TESTING_DEPLOYMENT_RULES.md
   ├─ Logging? → TESTING_DEPLOYMENT_RULES.md
   ├─ Monitor? → TESTING_DEPLOYMENT_RULES.md
   └─ Troubleshoot? → TESTING_DEPLOYMENT_RULES.md
```

---

## Tips for Efficient Usage

### 📌 Bookmark These
- [ ] README_RULES.md (for overview)
- [ ] QUICK_REFERENCE.md (for daily use)
- [ ] Your current task's document

### 💡 Pro Tips
1. **Keep documents in browser tabs** while working
2. **Copy code examples** - they're tested patterns
3. **Use browser search (Ctrl+F)** to find sections
4. **Follow checklists exactly** - they prevent issues
5. **Reference examples first** - then adapt to your needs

### ⏱️ Time Estimates
- Reading full document: 30-40 minutes
- Finding specific section: 2-3 minutes
- Copying code template: 5 minutes
- Running checklist: 10-15 minutes

### 🔄 Reuse Patterns
- **Docstring template** - Copy, fill in specifics
- **Test class template** - Copy, modify test names
- **Error handling pattern** - Copy, adjust error types
- **Data validation pattern** - Copy, adjust columns
- **Model metadata** - Copy, update values

---

## Troubleshooting: Can't Find Answer?

```
Lost? Try this sequence:

1. Look in QUICK_REFERENCE.md
   - Most common issues are there

2. Look in README_RULES.md
   - Common questions section
   - How to reference examples

3. Search the specific document
   - Use browser search (Ctrl+F)
   - Search for keywords

4. Check the index/table of contents
   - All major topics listed

5. Still stuck?
   - Check the "Common Mistakes" section
   - Review the decision tree above
   - Look at code examples
   - Follow the nearest checklist
```

---

## Document Relationships

```
README_RULES.md (overview & navigation)
    │
    ├─→ QUICK_REFERENCE.md (daily reference)
    │
    ├─→ PROJECT_RULES.md (structure & standards)
    │   └─→ General organization guidelines
    │
    ├─→ CODE_QUALITY_RULES.md (code style)
    │   └─→ How to write Python code
    │
    ├─→ DATA_MODEL_RULES.md (data & ML)
    │   ├─→ How to handle data
    │   └─→ How to train models
    │
    └─→ TESTING_DEPLOYMENT_RULES.md (QA & deployment)
        ├─→ How to write tests
        └─→ How to deploy

All documents cross-reference each other
with "See document_name.md - Section"
```

---

## Quick Verification: Are You Following Rules?

### ✅ Code Review Checklist (2 min)

```
□ File in correct location? (check PROJECT_RULES.md)
□ File named correctly? (check CODE_QUALITY_RULES.md)
□ Type hints added? (check CODE_QUALITY_RULES.md)
□ Docstrings complete? (check examples)
□ < 50 lines per function? (CODE_QUALITY_RULES.md)
□ Uses logging not print? (PROJECT_RULES.md)
□ No hardcoded paths? (PROJECT_RULES.md)
□ Inputs validated? (CODE_QUALITY_RULES.md)
□ Error handling? (CODE_QUALITY_RULES.md)
□ Tests written? (TESTING_DEPLOYMENT_RULES.md)
□ All tests pass? (TESTING_DEPLOYMENT_RULES.md)
□ Coverage 80%+? (TESTING_DEPLOYMENT_RULES.md)
```

### ✅ Data Review Checklist (2 min)

```
□ Raw data not modified? (DATA_MODEL_RULES.md)
□ Processed in data/processed/? (PROJECT_RULES.md)
□ Metadata saved? (DATA_MODEL_RULES.md)
□ Data validated? (DATA_MODEL_RULES.md)
□ Processing steps documented? (DATA_MODEL_RULES.md)
□ Versioned with date? (DATA_MODEL_RULES.md)
```

### ✅ Model Review Checklist (2 min)

```
□ Seeds set? (DATA_MODEL_RULES.md)
□ Proper split strategy? (DATA_MODEL_RULES.md)
□ Test data not in training? (DATA_MODEL_RULES.md)
□ Metadata saved? (DATA_MODEL_RULES.md)
□ Correctly named? (DATA_MODEL_RULES.md)
□ Metrics logged? (DATA_MODEL_RULES.md)
```

---

**Created:** December 2025
**Version:** 1.0
**Use this with the other rule documents for best results!**
