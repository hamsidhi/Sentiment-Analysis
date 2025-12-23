"""
Model Training and Evaluation Script for Sentiment Analysis
===========================================================

This script provides a complete pipeline for training, evaluating, and comparing
multiple machine learning models for sentiment analysis.

Features:
---------
- Train multiple models with a single command
- Detailed evaluation metrics and visualizations
- Model comparison and selection
- Cross-validation for robust evaluation
- Automated hyperparameter tuning (optional)
- Save best models for deployment

Author: AI Assistant
Python Version: 3.12+
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
import pickle
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from .sentiment_analyzer import SentimentAnalyzer
from .text_preprocessor import TextPreprocessor


class ModelTrainer:
    """
    Comprehensive model training and evaluation class for sentiment analysis.
    
    This class handles the entire machine learning workflow from data preparation
    to model selection and evaluation.
    """
    
    def __init__(self, max_features: int = 5000, random_state: int = 42):
        """
        Initialize the ModelTrainer.
        
        Parameters:
        -----------
        max_features : int
            Maximum number of features for TF-IDF vectorization
        random_state : int
            Random seed for reproducible results
        """
        self.max_features = max_features
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        
        # Define available models with their configurations
        self.available_models = {
            'logistic_regression': {
                'name': 'Logistic Regression',
                'model': LogisticRegression(random_state=random_state, max_iter=1000),
                'description': 'Simple, fast, and interpretable. Good baseline model.'
            },
            'random_forest': {
                'name': 'Random Forest',
                'model': RandomForestClassifier(n_estimators=100, random_state=random_state),
                'description': 'Ensemble method using multiple decision trees.'
            },
            'svm': {
                'name': 'Support Vector Machine',
                'model': LinearSVC(random_state=random_state),
                'description': 'Finds optimal boundary between classes.'
            },
            'naive_bayes': {
                'name': 'Naive Bayes',
                'model': MultinomialNB(),
                'description': 'Fast probabilistic classifier good for text data.'
            },
            'gradient_boosting': {
                'name': 'Gradient Boosting',
                'model': GradientBoostingClassifier(random_state=random_state),
                'description': 'Builds models sequentially to correct errors.'
            }
        }
    
    def prepare_data(self, data: pd.DataFrame, text_column: str = 'text',
                    label_column: str = 'sentiment', test_size: float = 0.2,
                    preprocessing_method: str = 'basic') -> Tuple[np.ndarray, np.ndarray, 
                                                                   np.ndarray, np.ndarray]:
        """
        Prepare data for training by cleaning and splitting.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw dataset with text and labels
        text_column : str
            Name of the column containing text
        label_column : str
            Name of the column containing labels
        test_size : float
            Proportion of data to use for testing
        preprocessing_method : str
            Method of preprocessing ('basic' or 'advanced')
            
        Returns:
        --------
        Tuple of (X_train, X_test, y_train, y_test)
            Split and preprocessed data ready for training
        """
        print("Preparing data for training...")
        print("-" * 40)
        
        # Initialize preprocessor
        preprocessor = TextPreprocessor()
        
        # Clean the text data
        print(f"Cleaning text data using {preprocessing_method} method...")
        if preprocessing_method == 'basic':
            data['cleaned_text'] = data[text_column].apply(preprocessor.basic_clean)
        else:
            data['cleaned_text'] = data[text_column].apply(preprocessor.advanced_clean)
        
        # Remove any empty texts after cleaning
        data = data[data['cleaned_text'].str.len() > 0]
        
        # Prepare features and labels
        X = data['cleaned_text']
        y = data[label_column]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"✓ Training set size: {len(X_train)}")
        print(f"✓ Test set size: {len(X_test)}")
        print(f"✓ Total samples after cleaning: {len(data)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_single_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, any]:
        """
        Train and evaluate a single model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to train
        X_train, y_train : array-like
            Training data
        X_test, y_test : array-like
            Test data
            
        Returns:
        --------
        Dict containing training results and metrics
        """
        print(f"\nTraining {model_name}...")
        print("-" * 40)
        
        start_time = time.time()
        
        # Get model configuration
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available. "
                           f"Choose from: {list(self.available_models.keys())}")
        
        model_config = self.available_models[model_name]
        model = model_config['model']
        
        # Create pipeline with TF-IDF vectorizer and the model
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english',
                lowercase=True,
                ngram_range=(1, 2)
            )),
            ('classifier', model)
        ])
        
        # Train the model
        print(f"Training {model_config['name']}...")
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        training_time = time.time() - start_time
        
        # Store results
        result = {
            'model_name': model_name,
            'pipeline': pipeline,
            'accuracy': accuracy,
            'predictions': y_pred,
            'training_time': training_time,
            'model_info': model_config
        }
        
        print(f"✓ Training completed in {training_time:.2f} seconds")
        print(f"✓ Accuracy: {accuracy:.2%}")
        
        return result
    
    def train_multiple_models(self, data: pd.DataFrame, 
                            models: List[str] = None,
                            text_column: str = 'text',
                            label_column: str = 'sentiment',
                            preprocessing_method: str = 'basic') -> pd.DataFrame:
        """
        Train and compare multiple models on the same dataset.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Training dataset
        models : List[str], optional
            List of model names to train. If None, trains all available models
        text_column : str
            Name of text column in dataset
        label_column : str
            Name of label column in dataset
        preprocessing_method : str
            Preprocessing method to use
            
        Returns:
        --------
        pd.DataFrame
            Comparison results with metrics for each model
        """
        print("Training Multiple Models")
        print("="*60)
        
        # Use all models if none specified
        if models is None:
            models = list(self.available_models.keys())
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(
            data, text_column, label_column, preprocessing_method=preprocessing_method
        )
        
        # Store test data for later use
        self.X_test = X_test
        self.y_test = y_test
        
        # Train each model
        results_list = []
        self.results = {}
        
        for model_name in models:
            try:
                result = self.train_single_model(model_name, X_train, y_train, X_test, y_test)
                
                # Store result
                self.results[model_name] = result
                
                # Add to comparison list
                results_list.append({
                    'Model': self.available_models[model_name]['name'],
                    'Accuracy': f"{result['accuracy']:.2%}",
                    'Training Time (s)': f"{result['training_time']:.2f}",
                    'Rank': 0  # Will be filled after sorting
                })
                
            except Exception as e:
                print(f"⚠ Error training {model_name}: {str(e)}")
                continue
        
        # Create comparison DataFrame
        results_df = pd.DataFrame(results_list)
        
        # Sort by accuracy and add ranks
        results_df['Accuracy_Numeric'] = results_df['Accuracy'].str.rstrip('%').astype(float)
        results_df = results_df.sort_values('Accuracy_Numeric', ascending=False)
        results_df['Rank'] = range(1, len(results_df) + 1)
        results_df = results_df.drop('Accuracy_Numeric', axis=1)
        
        # Store best model
        if not results_df.empty:
            best_model_name = list(self.results.keys())[0]  # First in sorted list
            self.best_model = self.results[best_model_name]['pipeline']
            self.best_score = self.results[best_model_name]['accuracy']
        
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        print(results_df.to_string(index=False))
        
        return results_df
    
    def detailed_evaluation(self, model_name: str = None) -> None:
        """
        Show detailed evaluation for a specific model or the best model.
        
        Parameters:
        -----------
        model_name : str, optional
            Name of the model to evaluate. If None, evaluates the best model
        """
        if not self.results:
            raise ValueError("No models trained yet. Call train_multiple_models() first.")
        
        # Use best model if none specified
        if model_name is None:
            model_name = max(self.results.keys(), 
                           key=lambda x: self.results[x]['accuracy'])
        
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        result = self.results[model_name]
        y_pred = result['predictions']
        
        print(f"\nDetailed Evaluation for {result['model_info']['name']}")
        print("="*60)
        
        # Classification report
        print("Classification Report:")
        print("-" * 30)
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {result["model_info"]["name"]}')
        plt.xlabel('Predicted Sentiment')
        plt.ylabel('Actual Sentiment')
        plt.tight_layout()
        plt.show()
        
        print("\nConfusion Matrix:")
        print(f"True Negatives:  {cm[0, 0]} (correctly identified negative)")
        print(f"False Positives: {cm[0, 1]} (negative classified as positive)")
        print(f"False Negatives: {cm[1, 0]} (positive classified as negative)")
        print(f"True Positives:  {cm[1, 1]} (correctly identified positive)")
    
    def visualize_model_comparison(self, save_path: Optional[str] = None):
        """
        Create visualizations comparing all trained models.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the visualization
        """
        if not self.results:
            raise ValueError("No models trained yet. Call train_multiple_models() first.")
        
        # Create comparison visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Accuracy comparison bar chart
        model_names = [self.results[key]['model_info']['name'] for key in self.results.keys()]
        accuracies = [self.results[key]['accuracy'] for key in self.results.keys()]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
        bars = axes[0].bar(model_names, accuracies, color=colors, alpha=0.7)
        axes[0].set_title('Model Accuracy Comparison')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')
        
        axes[0].grid(True, alpha=0.3)
        
        # 2. Training time comparison
        training_times = [self.results[key]['training_time'] for key in self.results.keys()]
        
        bars2 = axes[1].bar(model_names, training_times, color=colors, alpha=0.7)
        axes[1].set_title('Training Time Comparison')
        axes[1].set_ylabel('Time (seconds)')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, time in zip(bars2, training_times):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Model comparison saved to {save_path}")
        
        plt.show()
    
    def save_best_model(self, filepath: str):
        """
        Save the best performing model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path where the model should be saved
        """
        if self.best_model is None:
            raise ValueError("No models trained yet. Call train_multiple_models() first.")
        
        # Save the model pipeline
        with open(filepath, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        print(f"✓ Best model saved to {filepath}")
        print(f"✓ Model accuracy: {self.best_score:.2%}")
    
    def cross_validate_model(self, model_name: str, data: pd.DataFrame, 
                           cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation for more robust model evaluation.
        
        Cross-validation gives a more reliable estimate of model performance
        by testing on multiple subsets of the data.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to cross-validate
        data : pd.DataFrame
            Dataset for cross-validation
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        Dict[str, float]
            Cross-validation results
        """
        print(f"\nPerforming {cv_folds}-fold cross-validation for {model_name}...")
        print("-" * 60)
        
        # Prepare data
        preprocessor = TextPreprocessor()
        data['cleaned_text'] = data.iloc[:, 0].apply(preprocessor.basic_clean)
        data = data[data['cleaned_text'].str.len() > 0]
        
        X = data['cleaned_text']
        y = data.iloc[:, 1]
        
        # Get model
        model_config = self.available_models[model_name]
        model = model_config['model']
        
        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english',
                lowercase=True
            )),
            ('classifier', model)
        ])
        
        # Perform cross-validation
        cv_scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring='accuracy')
        
        results = {
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'min_accuracy': cv_scores.min(),
            'max_accuracy': cv_scores.max()
        }
        
        print(f"✓ Mean Accuracy: {results['mean_accuracy']:.2%} (+/- {results['std_accuracy']:.2%})")
        print(f"✓ Range: {results['min_accuracy']:.2%} to {results['max_accuracy']:.2%}")
        
        return results


def main():
    """
    Example usage of the ModelTrainer class.
    
    This function demonstrates how to use the training pipeline.
    """
    print("Model Training Script - Example Usage")
    print("="*50)
    
    # Example: Create sample data for demonstration
    sample_data = pd.DataFrame({
        'text': [
            "This movie was amazing! I loved it.",
            "Terrible film. Complete waste of time.",
            "Great acting and wonderful story.",
            "Boring and predictable plot.",
            "Excellent cinematography and music.",
            "Poor character development.",
            "Fantastic special effects!",
            "Disappointing ending.",
            "Brilliant performances by all actors.",
            "Awful dialogue and weak story."
        ],
        'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive',
                     'negative', 'positive', 'negative', 'positive', 'negative']
    })
    
    print("\n1. Available Models:")
    print("-" * 30)
    trainer = ModelTrainer()
    for key, config in trainer.available_models.items():
        print(f"• {key}: {config['name']}")
        print(f"  {config['description']}")
        print()
    
    print("\n2. Training Process:")
    print("-" * 30)
    print("To train models with your data:")
    print(">>> trainer = ModelTrainer()")
    print(">>> results = trainer.train_multiple_models(your_data)")
    print(">>> trainer.visualize_model_comparison()")
    print(">>> trainer.save_best_model('best_model.pkl')")
    
    print("\n3. Model Comparison Features:")
    print("-" * 30)
    print("• Train multiple models simultaneously")
    print("• Compare accuracy and training time")
    print("• Detailed evaluation reports")
    print("• Cross-validation for robust testing")
    print("• Save best performing model")
    
    print("\n" + "="*50)
    print("ModelTrainer is ready to use!")
    print("="*50)


if __name__ == "__main__":
    main()