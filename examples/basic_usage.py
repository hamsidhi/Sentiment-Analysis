"""
Basic Usage Examples for Sentiment Analysis Project
===================================================

This script demonstrates the most common use cases and features
of the sentiment analysis project in a simple, easy-to-understand way.

Perfect for beginners and quick reference!
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sentiment_analyzer import SentimentAnalyzer
from src.text_preprocessor import TextPreprocessor
from src.train_models import ModelTrainer
from src.predict import SentimentPredictor


def example_1_simple_sentiment_analysis():
    """
    Example 1: Simple sentiment analysis with minimal code.
    
    This is the quickest way to get started with sentiment analysis.
    """
    print("Example 1: Simple Sentiment Analysis")
    print("=" * 50)
    
    # Create some sample data (you would load real data in practice)
    sample_data = pd.DataFrame({
        'text': [
            "I love this movie! It's amazing!",
            "This film is terrible and boring.",
            "Great acting and wonderful story.",
            "Worst movie I've ever seen.",
            "Excellent cinematography and music.",
            "Poor plot and weak characters.",
            "Fantastic special effects!",
            "Disappointing and slow.",
            "Brilliant performances throughout.",
            "Awful dialogue and story."
        ],
        'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive',
                     'negative', 'positive', 'negative', 'positive', 'negative']
    })
    
    # Step 1: Create analyzer
    analyzer = SentimentAnalyzer(model_type='logistic_regression')
    
    # Step 2: Train the model
    print("Training model...")
    metrics = analyzer.train(sample_data)
    print(f"✓ Model trained! Accuracy: {metrics['accuracy']:.2%}")
    
    # Step 3: Make predictions
    test_texts = [
        "This movie is absolutely fantastic!",
        "I didn't enjoy this film at all.",
        "It was okay, nothing special."
    ]
    
    print("\nMaking predictions:")
    for text in test_texts:
        result = analyzer.predict(text)
        print(f"Text: '{text}'")
        print(f"Sentiment: {result}\n")
    
    return analyzer


def example_2_model_comparison():
    """
    Example 2: Compare different machine learning models.
    
    See which model works best for your data!
    """
    print("\nExample 2: Model Comparison")
    print("=" * 50)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'review': [
            "Excellent product, highly recommend!",
            "Poor quality, waste of money.",
            "Fast delivery and great packaging.",
            "Item arrived damaged, very disappointed.",
            "Perfect condition, exactly as described.",
            "Terrible customer service experience.",
            "Good value for the price.",
            "Not worth the money.",
            "Amazing quality and fast shipping!",
            "Defective product, returning immediately."
        ],
        'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive',
                     'negative', 'positive', 'negative', 'positive', 'negative']
    })
    
    # Create trainer
    trainer = ModelTrainer()
    
    # Compare models
    print("Comparing multiple models...")
    results = trainer.train_multiple_models(
        sample_data,
        text_column='review',
        models=['logistic_regression', 'naive_bayes', 'svm']
    )
    
    print("\nBest model:", results.iloc[0]['Model'])
    print("Best accuracy:", results.iloc[0]['Accuracy'])
    
    # Save best model
    trainer.save_best_model('best_model.pkl')
    print("✓ Best model saved!")
    
    return trainer


def example_3_text_preprocessing():
    """
    Example 3: Advanced text preprocessing and analysis.
    
    Learn how to clean and analyze your text data.
    """
    print("\nExample 3: Text Preprocessing")
    print("=" * 50)
    
    # Sample messy text data
    messy_texts = [
        "This movie was AMAZING!!! 😍😍😍",
        "Check out this review at https://example.com @user #movie",
        "Terrible film... really bad acting and plot!!!",
        "GREAT PRODUCT!!! RECOMMEND TO EVERYONE!!!",
        "The movie was okay, nothing special tbh",
        "",
        "Hi",
        "I absolutely love this product! It's the best purchase I've ever made!"
    ]
    
    # Create preprocessor
    preprocessor = TextPreprocessor(
        remove_stopwords=True,
        handle_emojis=True
    )
    
    print("Original vs Cleaned text:")
    print("-" * 30)
    for i, text in enumerate(messy_texts[:3], 1):
        cleaned = preprocessor.basic_clean(text)
        print(f"Original {i}: {text}")
        print(f"Cleaned  {i}: {cleaned}\n")
    
    # Get text statistics
    print("Text Statistics:")
    print("-" * 30)
    stats = preprocessor.get_text_statistics(messy_texts)
    print(f"Total texts: {stats['total_texts']}")
    print(f"Average length: {stats['avg_length']:.1f} characters")
    print(f"Unique words: {stats['unique_words']}")
    
    # Detect issues
    print("\nText Quality Issues:")
    print("-" * 30)
    issues = preprocessor.detect_language_issues(messy_texts)
    preprocessor.print_issues_summary(issues)
    
    return preprocessor


def example_4_batch_predictions():
    """
    Example 4: Batch predictions and CSV processing.
    
    Process multiple texts or entire files efficiently.
    """
    print("\nExample 4: Batch Predictions")
    print("=" * 50)
    
    # Create sample CSV data
    sample_reviews = pd.DataFrame({
        'product_id': [1, 2, 3, 4, 5],
        'review_text': [
            "Excellent quality and fast delivery!",
            "Poor packaging, item arrived damaged.",
            "Good value for money, satisfied.",
            "Terrible customer service experience.",
            "Amazing product, exceeded expectations!"
        ]
    })
    
    # Save sample data
    sample_reviews.to_csv('sample_reviews.csv', index=False)
    print("✓ Sample CSV created: sample_reviews.csv")
    
    # Train a model first (using previous example data)
    sample_data = pd.DataFrame({
        'text': [
            "Excellent quality and fast delivery!",
            "Poor packaging, item arrived damaged.",
            "Good value for money, satisfied.",
            "Terrible customer service experience.",
            "Amazing product, exceeded expectations!"
        ],
        'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive']
    })
    
    analyzer = SentimentAnalyzer()
    analyzer.train(sample_data)
    analyzer.save_model('batch_prediction_model.pkl')
    
    # Create predictor
    predictor = SentimentPredictor('batch_prediction_model.pkl')
    
    # Method 1: Batch prediction from list
    print("\nBatch prediction from list:")
    texts = [
        "Great product, highly recommend!",
        "Poor quality, not worth the price.",
        "Fast shipping and good packaging."
    ]
    results = predictor.predict_batch(texts)
    
    for text, sentiment in zip(texts, results):
        print(f"'{text[:40]}...' -> {sentiment}")
    
    # Method 2: Process CSV file
    print("\nProcessing CSV file:")
    results_df = predictor.predict_from_csv(
        'sample_reviews.csv',
        'review_text',
        'reviews_with_sentiment.csv'
    )
    
    print("Results:")
    print(results_df[['product_id', 'review_text', 'sentiment']])
    
    # Method 3: Get prediction summary
    print("\nPrediction Summary:")
    summary = predictor.get_prediction_summary('sample_reviews.csv', 'review_text')
    print(f"Total: {summary['total_predictions']}")
    print(f"Positive: {summary['positive_count']} ({summary['positive_percentage']:.1f}%)")
    print(f"Negative: {summary['negative_count']} ({summary['negative_percentage']:.1f}%)")
    
    return predictor


def example_5_confidence_scores():
    """
    Example 5: Get confidence scores with predictions.
    
    Understand how confident the model is in its predictions.
    """
    print("\nExample 5: Confidence Scores")
    print("=" * 50)
    
    # Train model with example data
    sample_data = pd.DataFrame({
        'text': [
            "This product is absolutely amazing! Best purchase ever!",
            "Terrible quality, complete waste of money.",
            "The product is okay, nothing special.",
            "Excellent service and fast delivery!",
            "Poor customer support, very disappointed."
        ],
        'sentiment': ['positive', 'negative', 'positive', 'positive', 'negative']
    })
    
    analyzer = SentimentAnalyzer(model_type='logistic_regression')
    analyzer.train(sample_data)
    
    # Test texts with varying clarity
    test_texts = [
        "This is absolutely fantastic and amazing!",
        "This is terrible and awful!",
        "It is okay, not bad but not great either.",
        "The product is fine I guess.",
        "Worst purchase ever, completely useless!"
    ]
    
    print("Predictions with confidence scores:")
    print("-" * 40)
    
    for text in test_texts:
        # Get confidence scores
        confidence = analyzer.predict(text, return_confidence=True)
        
        if isinstance(confidence, dict):
            positive_conf = confidence.get('positive', 0) * 100
            negative_conf = confidence.get('negative', 0) * 100
            
            if positive_conf > negative_conf:
                sentiment = 'positive'
                conf_score = positive_conf
            else:
                sentiment = 'negative'
                conf_score = negative_conf
            
            print(f"Text: '{text}'")
            print(f"Sentiment: {sentiment} (confidence: {conf_score:.1f}%)")
            print(f"  Positive: {positive_conf:.1f}%")
            print(f"  Negative: {negative_conf:.1f}%\n")
        else:
            print(f"Text: '{text}'")
            print(f"Sentiment: {confidence}\n")
    
    return analyzer


def example_6_custom_preprocessing():
    """
    Example 6: Custom preprocessing for specific use cases.
    
    Adapt preprocessing for different types of text data.
    """
    print("\nExample 6: Custom Preprocessing")
    print("=" * 50)
    
    # Different types of text data
    social_media_texts = [
        "OMG this movie is AMAZING!!! 😍😍 @friend #BestMovieEver",
        "Can't believe how bad this was... 😡 #WasteOfTime",
        "RT @user: This film is absolutely incredible!",
        "Check out my review: https://example.com/movie-review"
    ]
    
    formal_reviews = [
        "The cinematography was excellent and the acting was superb.",
        "The plot lacked coherence and character development was poor.",
        "A masterful piece of filmmaking with outstanding performances."
    ]
    
    # Preprocessor for social media
    print("Social Media Preprocessing:")
    print("-" * 30)
    social_preprocessor = TextPreprocessor(
        remove_stopwords=True,
        handle_emojis=True,
        correct_spelling=False  # Don't correct social media slang
    )
    
    for text in social_media_texts:
        cleaned = social_preprocessor.advanced_clean(text)
        print(f"Original: {text}")
        print(f"Cleaned:  {cleaned}\n")
    
    # Preprocessor for formal reviews
    print("\nFormal Review Preprocessing:")
    print("-" * 30)
    formal_preprocessor = TextPreprocessor(
        remove_stopwords=False,  # Keep formal language
        handle_emojis=False,     # No emojis in formal text
        correct_spelling=True    # Correct any typos
    )
    
    for text in formal_reviews:
        cleaned = formal_preprocessor.basic_clean(text)
        print(f"Original: {text}")
        print(f"Cleaned:  {cleaned}\n")
    
    return social_preprocessor, formal_preprocessor


def run_all_examples():
    """
    Run all examples in sequence.
    
    This demonstrates the complete workflow of the sentiment analysis project.
    """
    print("SENTIMENT ANALYSIS PROJECT - COMPLETE EXAMPLES")
    print("=" * 60)
    print("This script demonstrates all major features of the project.\n")
    
    try:
        # Run all examples
        analyzer1 = example_1_simple_sentiment_analysis()
        trainer = example_2_model_comparison()
        preprocessor = example_3_text_preprocessing()
        predictor = example_4_batch_predictions()
        analyzer2 = example_5_confidence_scores()
        social_prep, formal_prep = example_6_custom_preprocessing()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nYou now know how to:")
        print("✓ Create and train sentiment analysis models")
        print("✓ Compare different machine learning models")
        print("✓ Preprocess and clean text data")
        print("✓ Make predictions on single texts and batches")
        print("✓ Get confidence scores for predictions")
        print("✓ Customize preprocessing for different text types")
        print("\nReady to use with your own datasets!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("Don't worry! This is normal when learning.")
        print("Check the error message and try the specific example again.")


if __name__ == "__main__":
    run_all_examples()