"""
Sentiment Analysis Project Package
==================================

A comprehensive sentiment analysis toolkit for learning and practical applications.

Features:
---------
- Easy-to-use sentiment analysis with multiple ML models
- Advanced text preprocessing capabilities
- Model training and evaluation tools
- Batch prediction and interactive mode
- Support for various datasets (IMDB, Twitter, Amazon reviews)

Quick Start:
------------
>>> from src.sentiment_analyzer import SentimentAnalyzer
>>> analyzer = SentimentAnalyzer()
>>> # Load data and train model
>>> # Make predictions on new text

Author: AI Assistant
Version: 1.0.0
Python: 3.12+
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = "your.email@example.com"

# Import main classes for easy access
from .src.sentiment_analyzer import SentimentAnalyzer
from .src.text_preprocessor import TextPreprocessor
from .src.train_models import ModelTrainer
from .src.predict import SentimentPredictor

__all__ = [
    "SentimentAnalyzer",
    "TextPreprocessor", 
    "ModelTrainer",
    "SentimentPredictor"
]