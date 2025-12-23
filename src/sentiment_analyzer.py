"""
Sentiment Analysis Project - Main Module
========================================

This is the main sentiment analysis module that provides a complete pipeline
for training, evaluating, and using sentiment analysis models.

Features:
---------
- Easy to use with any text dataset
- Supports multiple machine learning models
- Detailed comments for learning
- Error handling and validation
- Model persistence for future use

Author: AI Assistant
Python Version: 3.12+
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle
import warnings
import re
import string
from typing import List, Tuple, Dict, Union, Optional

# Filter out warnings to keep output clean
warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """
    A comprehensive sentiment analysis class that handles the entire ML pipeline.
    
    This class provides methods for:
    - Data loading and preprocessing
    - Model training and evaluation
    - Prediction on new text
    - Model saving and loading
    """
    
    def __init__(self, model_type: str = 'logistic_regression', max_features: int = 5000):
        """
        Initialize the SentimentAnalyzer.
        
        Parameters:
        -----------
        model_type : str
            The type of machine learning model to use.
            Options: 'logistic_regression', 'random_forest', 'svm', 'naive_bayes'
        max_features : int
            Maximum number of features for the TF-IDF vectorizer
            
        Example:
        --------
        >>> analyzer = SentimentAnalyzer(model_type='logistic_regression')
        """
        self.model_type = model_type
        self.max_features = max_features
        self.model = None
        self.vectorizer = None
        self.model_pipeline = None
        self.is_trained = False
        
        # Initialize the model based on the specified type
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Private method to initialize the machine learning model.
        This method sets up the model pipeline based on the chosen algorithm.
        """
        # Create TF-IDF vectorizer to convert text to numerical features
        # TF-IDF stands for Term Frequency-Inverse Document Frequency
        # It helps identify important words in documents
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',  # Remove common English words like 'the', 'and', etc.
            lowercase=True,        # Convert all text to lowercase
            ngram_range=(1, 2)     # Use both single words and pairs of words (bigrams)
        )
        
        # Select and initialize the machine learning model
        if self.model_type == 'logistic_regression':
            # Logistic Regression is simple, fast, and works well for text classification
            self.model = LogisticRegression(random_state=42, max_iter=1000)
            
        elif self.model_type == 'random_forest':
            # Random Forest uses multiple decision trees for better accuracy
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            
        elif self.model_type == 'svm':
            # Support Vector Machine finds the best boundary between classes
            self.model = LinearSVC(random_state=42)
            
        elif self.model_type == 'naive_bayes':
            # Naive Bayes is fast and works well with text data
            # It's based on probability and assumes features are independent
            self.model = MultinomialNB()
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. "
                           f"Choose from: logistic_regression, random_forest, svm, naive_bayes")
        
        # Create a pipeline that first vectorizes text, then applies the model
        # A pipeline chains multiple steps together
        self.model_pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.model)
        ])
    
    def load_data(self, file_path: str, text_column: str = 'text', 
                  label_column: str = 'sentiment') -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        This method is flexible and can work with any CSV file that has text and label columns.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing the data
        text_column : str
            Name of the column containing the text/reviews
        label_column : str
            Name of the column containing the sentiment labels
            
        Returns:
        --------
        pd.DataFrame
            The loaded dataset with text and labels
            
        Example:
        --------
        >>> # For IMDB dataset with columns 'review' and 'sentiment'
        >>> data = analyzer.load_data('imdb_data.csv', text_column='review', label_column='sentiment')
        >>> 
        >>> # For Twitter dataset with columns 'tweet' and 'label'
        >>> data = analyzer.load_data('twitter_data.csv', text_column='tweet', label_column='label')
        """
        try:
            # Try to load the data
            data = pd.read_csv(file_path)
            print(f"✓ Data loaded successfully from {file_path}")
            print(f"✓ Dataset shape: {data.shape}")
            print(f"✓ Columns found: {list(data.columns)}")
            
            # Check if the specified columns exist
            if text_column not in data.columns:
                raise ValueError(f"Text column '{text_column}' not found. "
                               f"Available columns: {list(data.columns)}")
            
            if label_column not in data.columns:
                raise ValueError(f"Label column '{label_column}' not found. "
                               f"Available columns: {list(data.columns)}")
            
            # Select only the columns we need and rename them for consistency
            data = data[[text_column, label_column]].copy()
            data.columns = ['text', 'sentiment']
            
            # Remove any rows with missing values
            data = data.dropna()
            
            print(f"✓ Data prepared successfully. {len(data)} samples ready for training.")
            
            # Show some basic statistics about the data
            self._show_data_summary(data)
            
            return data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}. Please check the file path.")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def _show_data_summary(self, data: pd.DataFrame):
        """
        Private method to show summary statistics of the loaded data.
        
        This helps users understand their dataset better.
        """
        print("\n" + "="*50)
        print("DATASET SUMMARY")
        print("="*50)
        print(f"Total samples: {len(data)}")
        print(f"Text column length statistics:")
        text_lengths = data['text'].str.len()
        print(f"  - Average length: {text_lengths.mean():.1f} characters")
        print(f"  - Shortest text: {text_lengths.min()} characters")
        print(f"  - Longest text: {text_lengths.max()} characters")
        
        print(f"\nSentiment distribution:")
        sentiment_counts = data['sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(data)) * 100
            print(f"  - {sentiment}: {count} samples ({percentage:.1f}%)")
        print("="*50 + "\n")
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess a single text string.
        
        This method performs several cleaning operations:
        - Converts to lowercase
        - Removes special characters and numbers
        - Removes extra whitespace
        
        Parameters:
        -----------
        text : str
            The raw text to clean
            
        Returns:
        --------
        str
            The cleaned text
            
        Example:
        --------
        >>> cleaned = analyzer.preprocess_text("This movie was AMAZING!!! 😍")
        >>> print(cleaned)
        'this movie was amazing'
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase - makes the model case-insensitive
        text = str(text).lower()
        
        # Remove URLs - they don't contribute to sentiment
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (for social media text)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove numbers and special characters, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace and strip leading/trailing spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def train(self, data: pd.DataFrame, test_size: float = 0.2, 
              random_state: int = 42) -> Dict[str, float]:
        """
        Train the sentiment analysis model on the provided data.
        
        This method:
        1. Preprocesses the text data
        2. Splits data into training and testing sets
        3. Trains the model pipeline
        4. Evaluates the model performance
        
        Parameters:
        -----------
        data : pd.DataFrame
            The training data with 'text' and 'sentiment' columns
        test_size : float
            Proportion of data to use for testing (default: 0.2 = 20%)
        random_state : int
            Random seed for reproducible results
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing evaluation metrics
            
        Example:
        --------
        >>> analyzer = SentimentAnalyzer()
        >>> data = analyzer.load_data('my_data.csv')
        >>> metrics = analyzer.train(data)
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
        """
        print("Starting model training...")
        print("="*50)
        
        # Step 1: Preprocess the text data
        print("Step 1: Preprocessing text data...")
        data['cleaned_text'] = data['text'].apply(self.preprocess_text)
        print(f"✓ Preprocessed {len(data)} text samples")
        
        # Step 2: Split data into training and testing sets
        print("\nStep 2: Splitting data...")
        X = data['cleaned_text']  # Features (text)
        y = data['sentiment']     # Labels (sentiment)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"✓ Training set: {len(X_train)} samples")
        print(f"✓ Testing set: {len(X_test)} samples")
        
        # Step 3: Train the model
        print(f"\nStep 3: Training {self.model_type} model...")
        print("This may take a few minutes depending on dataset size...")
        
        self.model_pipeline.fit(X_train, y_train)
        print("✓ Model training completed!")
        
        # Step 4: Make predictions on test set
        print("\nStep 4: Evaluating model...")
        y_pred = self.model_pipeline.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✓ Model Accuracy: {accuracy:.2%}")
        
        # Store the evaluation results
        self.evaluation_results = {
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        # Mark model as trained
        self.is_trained = True
        
        print("="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        return {'accuracy': accuracy}
    
    def evaluate(self) -> None:
        """
        Display detailed evaluation metrics and visualizations.
        
        This method shows:
        - Classification report (precision, recall, F1-score)
        - Confusion matrix heatmap
        
        Should be called after training the model.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation. Call train() first.")
        
        print("\nDetailed Model Evaluation")
        print("="*50)
        
        # Get the stored results
        y_test = self.evaluation_results['y_test']
        y_pred = self.evaluation_results['y_pred']
        
        # Print classification report
        print("Classification Report:")
        print("-" * 30)
        print(classification_report(y_test, y_pred))
        
        # Create confusion matrix visualization
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix - Model Performance')
        plt.xlabel('Predicted Sentiment')
        plt.ylabel('Actual Sentiment')
        plt.tight_layout()
        plt.show()
        
        print("\nConfusion Matrix Explanation:")
        print("- Top-left: True Negatives (correctly identified negative reviews)")
        print("- Top-right: False Positives (negative reviews classified as positive)")
        print("- Bottom-left: False Negatives (positive reviews classified as negative)")
        print("- Bottom-right: True Positives (correctly identified positive reviews)")
    
    def predict(self, texts: Union[str, List[str]], return_confidence: bool = False) -> Union[str, List[str], Dict[str, float], List[Dict[str, float]]]:
        """
        Predict sentiment for new text(s).
        
        This method can handle both single strings and lists of strings.
        
        Parameters:
        -----------
        texts : str or List[str]
            The text(s) to analyze
        return_confidence : bool
            Whether to return confidence scores along with predictions
            
        Returns:
        --------
        str or List[str] or Dict[str, float] or List[Dict[str, float]]
            Predicted sentiment(s) - with confidence if requested
            
        Example:
        --------
        >>> # Single prediction
        >>> result = analyzer.predict("This movie was amazing!")
        >>> print(f"Sentiment: {result}")
        >>> 
        >>> # Multiple predictions
        >>> results = analyzer.predict(["Great product!", "Terrible service"])
        >>> for text, sentiment in zip(texts, results):
        ...     print(f"'{text}' -> {sentiment}")
        >>> 
        >>> # With confidence
        >>> result = analyzer.predict("This is great!", return_confidence=True)
        >>> print(result)  # {'positive': 0.85, 'negative': 0.15}
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions. Call train() first.")
        
        # Handle single string input
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        # Make predictions
        predictions = self.model_pipeline.predict(texts)
        
        if return_confidence and hasattr(self.model_pipeline, 'predict_proba'):
            # Get probability predictions
            probabilities = self.model_pipeline.predict_proba(texts)
            classes = self.model_pipeline.classes_
            
            confidence_results = []
            for i, pred in enumerate(predictions):
                confidence_dict = {}
                for j, class_name in enumerate(classes):
                    readable_name = 'positive' if (isinstance(class_name, (int, float)) and class_name == 1) or class_name == 'pos' else 'negative'
                    confidence_dict[readable_name] = probabilities[i][j]
                confidence_results.append(confidence_dict)
            
            # Return single dict if input was single string
            return confidence_results[0] if is_single else confidence_results
        else:
            # Convert numeric predictions to readable labels if needed
            readable_predictions = []
            for pred in predictions:
                if isinstance(pred, (int, float)):
                    readable_predictions.append('positive' if pred == 1 else 'negative')
                else:
                    readable_predictions.append(str(pred))
            
            # Return single string if input was single string
            return readable_predictions[0] if is_single else readable_predictions
    
    def predict_with_probability(self, text: str) -> Dict[str, float]:
        """
        Predict sentiment with confidence probabilities.
        
        This method provides more detailed information about the prediction,
        including the confidence scores for each class.
        
        Parameters:
        -----------
        text : str
            The text to analyze
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with sentiments as keys and probabilities as values
            
        Example:
        --------
        >>> result = analyzer.predict_with_probability("This movie was okay")
        >>> print(result)
        {'negative': 0.45, 'positive': 0.55}
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        # Check if the model supports probability prediction
        if not hasattr(self.model_pipeline, 'predict_proba'):
            raise ValueError(f"{self.model_type} model doesn't support probability prediction. "
                           "Use predict() instead.")
        
        # Get probability predictions
        probabilities = self.model_pipeline.predict_proba([text])[0]
        
        # Create dictionary with class names and probabilities
        classes = self.model_pipeline.classes_
        result = {}
        for i, class_name in enumerate(classes):
            # Convert numeric classes to readable names
            readable_name = 'positive' if (isinstance(class_name, (int, float)) and class_name == 1) or class_name == 'pos' else 'negative'
            result[readable_name] = probabilities[i]
        
        return result
    
    def save_model(self, filepath: str):
        """
        Save the trained model to a file for later use.
        
        This allows you to train a model once and use it multiple times
        without retraining.
        
        Parameters:
        -----------
        filepath : str
            Path where the model should be saved (should end with .pkl)
            
        Example:
        --------
        >>> analyzer.save_model('my_sentiment_model.pkl')
        >>> # Later, you can load it with:
        >>> # analyzer.load_model('my_sentiment_model.pkl')
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model. Train the model first.")
        
        # Create a dictionary with all model components
        model_data = {
            'model_pipeline': self.model_pipeline,
            'model_type': self.model_type,
            'max_features': self.max_features,
            'is_trained': self.is_trained
        }
        
        # Save to file using pickle
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved successfully to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a previously saved model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model file (.pkl file)
            
        Example:
        --------
        >>> analyzer = SentimentAnalyzer()
        >>> analyzer.load_model('my_sentiment_model.pkl')
        >>> result = analyzer.predict("This is great!")
        """
        try:
            # Load the model data from file
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore all model components
            self.model_pipeline = model_data['model_pipeline']
            self.model_type = model_data['model_type']
            self.max_features = model_data['max_features']
            self.is_trained = model_data['is_trained']
            
            print(f"✓ Model loaded successfully from {filepath}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {filepath}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def compare_models(self, data: pd.DataFrame, models: List[str] = None) -> pd.DataFrame:
        """
        Compare different machine learning models on the same dataset.
        
        This method helps you choose the best model for your specific dataset
        by training and evaluating multiple models.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The training data with 'text' and 'sentiment' columns
        models : List[str], optional
            List of model types to compare
            Default: ['logistic_regression', 'random_forest', 'svm', 'naive_bayes']
            
        Returns:
        --------
        pd.DataFrame
            Comparison results with accuracy scores for each model
            
        Example:
        --------
        >>> results = analyzer.compare_models(data)
        >>> print(results)
        >>> # Results will show accuracy for each model type
        """
        if models is None:
            models = ['logistic_regression', 'random_forest', 'svm', 'naive_bayes']
        
        print("Comparing Multiple Models")
        print("="*50)
        
        results = []
        
        for model_type in models:
            print(f"\nTraining {model_type}...")
            
            # Create a new analyzer with this model type
            temp_analyzer = SentimentAnalyzer(model_type=model_type, 
                                            max_features=self.max_features)
            
            # Train the model
            metrics = temp_analyzer.train(data, test_size=0.2)
            
            # Store results
            results.append({
                'model': model_type,
                'accuracy': metrics['accuracy']
            })
            
            print(f"✓ {model_type}: {metrics['accuracy']:.2%} accuracy")
        
        # Create comparison dataframe
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('accuracy', ascending=False)
        
        print("\n" + "="*50)
        print("MODEL COMPARISON RESULTS")
        print("="*50)
        print(results_df.to_string(index=False))
        
        # Visualize results
        plt.figure(figsize=(10, 6))
        plt.bar(results_df['model'], results_df['accuracy'], color='skyblue', alpha=0.7)
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Model Type')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate(results_df['accuracy']):
            plt.text(i, v + 0.01, f'{v:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return results_df


def main():
    """
    Example usage of the SentimentAnalyzer class.
    
    This function demonstrates how to use the sentiment analysis pipeline
    with different datasets and scenarios.
    """
    print("Sentiment Analysis Project - Example Usage")
    print("="*60)
    
    # Example 1: Create a sentiment analyzer
    print("\n1. Creating Sentiment Analyzer...")
    analyzer = SentimentAnalyzer(model_type='logistic_regression')
    print("✓ Sentiment analyzer created successfully!")
    
    # Note: To use this with real data, you would do something like:
    """
    # Load your data (replace 'your_data.csv' with actual file)
    # data = analyzer.load_data('your_data.csv', text_column='review', label_column='sentiment')
    
    # Train the model
    # metrics = analyzer.train(data)
    
    # Make predictions
    # result = analyzer.predict("This product is amazing!")
    # print(f"Sentiment: {result}")
    """
    
    print("\n2. Example predictions without training:")
    print("(Note: These would work after training with real data)")
    print("- Text: 'I love this movie!' -> Predicted: positive")
    print("- Text: 'This product is terrible' -> Predicted: negative")
    
    print("\n3. Available methods:")
    print("- analyzer.load_data()     # Load dataset from CSV")
    print("- analyzer.train()         # Train the model")
    print("- analyzer.predict()       # Make predictions")
    print("- analyzer.evaluate()      # Show detailed evaluation")
    print("- analyzer.save_model()    # Save trained model")
    print("- analyzer.load_model()    # Load saved model")
    print("- analyzer.compare_models() # Compare different ML models")
    
    print("\n" + "="*60)
    print("Ready to use with your datasets!")
    print("="*60)


if __name__ == "__main__":
    main()