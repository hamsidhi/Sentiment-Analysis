"""
Text Preprocessing Module for Sentiment Analysis
================================================

This module provides advanced text preprocessing utilities that can be used
with any sentiment analysis project. It includes various cleaning operations,
text normalization, and feature extraction methods.

Features:
---------
- Advanced text cleaning (removing noise, normalizing text)
- Support for different text sources (social media, reviews, etc.)
- Emoji and emoticon handling
- Spelling correction (optional)
- Text statistics and visualization

Author: AI Assistant
Python Version: 3.12+
"""

import re
import string
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Optional imports for advanced features
try:
    from textblob import TextBlob  # For spelling correction
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

try:
    import emoji  # For emoji handling
    HAS_EMOJI = True
except ImportError:
    HAS_EMOJI = False


class TextPreprocessor:
    """
    Advanced text preprocessing class for sentiment analysis.
    
    This class provides comprehensive text cleaning and preprocessing methods
    that can be customized based on your specific needs and data source.
    """
    
    def __init__(self, remove_stopwords: bool = True, 
                 correct_spelling: bool = False,
                 handle_emojis: bool = True):
        """
        Initialize the TextPreprocessor with custom options.
        
        Parameters:
        -----------
        remove_stopwords : bool
            Whether to remove common stop words (the, and, or, etc.)
        correct_spelling : bool
            Whether to attempt spelling correction (requires textblob)
        handle_emojis : bool
            Whether to convert emojis to text descriptions
            
        Example:
        --------
        >>> # For social media text
        >>> preprocessor = TextPreprocessor(handle_emojis=True)
        >>> 
        >>> # For formal reviews
        >>> preprocessor = TextPreprocessor(remove_stopwords=True, 
        ...                                correct_spelling=True)
        """
        self.remove_stopwords = remove_stopwords
        self.correct_spelling = correct_spelling and HAS_TEXTBLOB
        self.handle_emojis = handle_emojis and HAS_EMOJI
        
        # English stop words (common words that don't add meaning)
        self.stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
            'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 
            'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
            'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
            'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
            'with', 'through', 'during', 'before', 'after', 'above', 'below', 
            'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
            'further', 'then', 'once', 'to', 'from'
        }
    
    def basic_clean(self, text: str) -> str:
        """
        Perform basic text cleaning operations.
        
        This is the most common cleaning that works for most datasets.
        
        Parameters:
        -----------
        text : str
            Raw text to clean
            
        Returns:
        --------
        str
            Cleaned text
            
        Example:
        --------
        >>> preprocessor = TextPreprocessor()
        >>> text = "This movie was AMAZING!!! 😍 https://example.com @user"
        >>> cleaned = preprocessor.basic_clean(text)
        >>> print(cleaned)
        'this movie was amazing'
        """
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags for social media text
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers and special characters, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def advanced_clean(self, text: str) -> str:
        """
        Perform advanced text cleaning with additional features.
        
        This method includes:
        - Basic cleaning
        - Stop word removal
        - Spelling correction (if enabled)
        - Emoji handling (if enabled)
        
        Parameters:
        -----------
        text : str
            Raw text to clean
            
        Returns:
        --------
        str
            Thoroughly cleaned text
            
        Example:
        --------
        >>> preprocessor = TextPreprocessor(remove_stopwords=True)
        >>> text = "This movie was absolutely amazing and I really loved it!!!"
        >>> cleaned = preprocessor.advanced_clean(text)
        >>> print(cleaned)
        'movie absolutely amazing really loved'
        """
        # Start with basic cleaning
        text = self.basic_clean(text)
        
        # Handle emojis if enabled
        if self.handle_emojis and HAS_EMOJI:
            # Convert emojis to text descriptions
            text = emoji.demojize(text)
            # Remove the emoji delimiters and make it readable
            text = text.replace(':', ' ').replace('_', ' ')
        
        # Remove stop words if enabled
        if self.remove_stopwords:
            words = text.split()
            filtered_words = [word for word in words if word not in self.stop_words]
            text = ' '.join(filtered_words)
        
        # Correct spelling if enabled
        if self.correct_spelling and HAS_TEXTBLOB:
            try:
                blob = TextBlob(text)
                text = str(blob.correct())
            except:
                # If correction fails, keep original text
                pass
        
        return text.strip()
    
    def clean_dataframe(self, df: pd.DataFrame, text_column: str,
                       method: str = 'basic') -> pd.DataFrame:
        """
        Clean text in an entire pandas DataFrame column.
        
        This is a convenient method to clean all text in a dataset at once.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing text data
        text_column : str
            Name of the column containing text to clean
        method : str
            Cleaning method to use: 'basic' or 'advanced'
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with cleaned text in a new column
            
        Example:
        --------
        >>> preprocessor = TextPreprocessor()
        >>> df = pd.DataFrame({'review': ['Great product!', 'Terrible service']})
        >>> cleaned_df = preprocessor.clean_dataframe(df, 'review', 'basic')
        >>> print(cleaned_df['review_cleaned'].head())
        """
        # Create a copy to avoid modifying original data
        df_clean = df.copy()
        
        # Choose cleaning method
        if method == 'basic':
            clean_func = self.basic_clean
        elif method == 'advanced':
            clean_func = self.advanced_clean
        else:
            raise ValueError(f"Unknown method: {method}. Use 'basic' or 'advanced'.")
        
        # Apply cleaning to the text column
        df_clean[f'{text_column}_cleaned'] = df_clean[text_column].apply(clean_func)
        
        # Remove any rows where cleaned text is empty
        df_clean = df_clean[df_clean[f'{text_column}_cleaned'].str.len() > 0]
        
        return df_clean
    
    def get_text_statistics(self, texts: List[str]) -> Dict[str, any]:
        """
        Calculate various statistics about a collection of texts.
        
        This helps understand your dataset better and can guide preprocessing decisions.
        
        Parameters:
        -----------
        texts : List[str]
            List of text strings to analyze
            
        Returns:
        --------
        Dict[str, any]
            Dictionary containing various text statistics
            
        Example:
        --------
        >>> preprocessor = TextPreprocessor()
        >>> texts = ["Great product!", "Terrible service", "It was okay"]
        >>> stats = preprocessor.get_text_statistics(texts)
        >>> print(f"Average length: {stats['avg_length']}")
        """
        if not texts:
            return {}
        
        # Convert to pandas Series for easier analysis
        text_series = pd.Series(texts)
        
        # Calculate basic statistics
        lengths = text_series.str.len()
        word_counts = text_series.str.split().str.len()
        
        # Count characters
        all_text = ' '.join(texts).lower()
        char_counts = Counter(all_text)
        
        # Count words
        all_words = []
        for text in texts:
            all_words.extend(str(text).lower().split())
        word_counts_dict = Counter(all_words)
        
        # Calculate statistics
        stats = {
            'total_texts': len(texts),
            'avg_length': lengths.mean(),
            'min_length': lengths.min(),
            'max_length': lengths.max(),
            'avg_words': word_counts.mean(),
            'min_words': word_counts.min(),
            'max_words': word_counts.max(),
            'most_common_chars': char_counts.most_common(10),
            'most_common_words': word_counts_dict.most_common(20),
            'unique_words': len(word_counts_dict),
            'total_words': sum(word_counts_dict.values())
        }
        
        return stats
    
    def visualize_text_statistics(self, texts: List[str], save_path: Optional[str] = None):
        """
        Create visualizations for text statistics.
        
        This method creates multiple plots to help understand the text data distribution.
        
        Parameters:
        -----------
        texts : List[str]
            List of text strings to visualize
        save_path : str, optional
            Path to save the visualization (if None, just displays)
            
        Example:
        --------
        >>> preprocessor = TextPreprocessor()
        >>> texts = ["Great product!", "Terrible service", "It was okay"]
        >>> preprocessor.visualize_text_statistics(texts, 'text_stats.png')
        """
        stats = self.get_text_statistics(texts)
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Text Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Text length distribution
        lengths = [len(text) for text in texts]
        axes[0, 0].hist(lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(stats['avg_length'], color='red', linestyle='--', 
                          label=f'Mean: {stats["avg_length"]:.1f}')
        axes[0, 0].set_title('Text Length Distribution')
        axes[0, 0].set_xlabel('Characters')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Word count distribution
        word_counts = [len(str(text).split()) for text in texts]
        axes[0, 1].hist(word_counts, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].axvline(stats['avg_words'], color='red', linestyle='--', 
                          label=f'Mean: {stats["avg_words"]:.1f}')
        axes[0, 1].set_title('Word Count Distribution')
        axes[0, 1].set_xlabel('Words')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Most common words
        common_words = stats['most_common_words'][:15]  # Top 15 words
        words = [word for word, count in common_words]
        counts = [count for word, count in common_words]
        
        axes[1, 0].barh(words, counts, color='coral', alpha=0.7)
        axes[1, 0].set_title('Top 15 Most Common Words')
        axes[1, 0].set_xlabel('Frequency')
        axes[1, 0].invert_yaxis()
        
        # 4. Summary statistics as text
        axes[1, 1].axis('off')
        summary_text = f"""
        Text Summary:
        
        Total texts: {stats['total_texts']:,}
        Unique words: {stats['unique_words']:,}
        Total words: {stats['total_words']:,}
        
        Length Statistics:
        • Average: {stats['avg_length']:.1f} characters
        • Shortest: {stats['min_length']} characters
        • Longest: {stats['max_length']} characters
        
        Word Statistics:
        • Average: {stats['avg_words']:.1f} words/text
        • Minimum: {stats['min_words']} words
        • Maximum: {stats['max_words']} words
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, 
                       verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        
        plt.show()
    
    def create_wordcloud(self, texts: List[str], save_path: Optional[str] = None):
        """
        Create a word cloud visualization from text data.
        
        Word clouds show the most frequent words in a visually appealing way.
        
        Parameters:
        -----------
        texts : List[str]
            List of text strings to visualize
        save_path : str, optional
            Path to save the word cloud (if None, just displays)
            
        Example:
        --------
        >>> preprocessor = TextPreprocessor()
        >>> reviews = ["Great product!", "Excellent quality", "Fast delivery"]
        >>> preprocessor.create_wordcloud(reviews, 'wordcloud.png')
        """
        try:
            from wordcloud import WordCloud
        except ImportError:
            print("⚠ WordCloud library not installed. Install with: pip install wordcloud")
            return
        
        # Combine all texts and clean them
        combined_text = ' '.join([self.basic_clean(text) for text in texts])
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            stopwords=self.stop_words if self.remove_stopwords else set(),
            max_words=100,
            colormap='viridis'
        ).generate(combined_text)
        
        # Display
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Words in Text Data', fontsize=16, pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Word cloud saved to {save_path}")
        
        plt.show()
    
    def detect_language_issues(self, texts: List[str]) -> Dict[str, List[str]]:
        """
        Detect potential issues in text data that might affect sentiment analysis.
        
        This method identifies various text quality issues that users should be aware of.
        
        Parameters:
        -----------
        texts : List[str]
            List of text strings to analyze
            
        Returns:
        --------
        Dict[str, List[str]]
            Dictionary categorizing different types of text issues
            
        Example:
        --------
        >>> preprocessor = TextPreprocessor()
        >>> issues = preprocessor.detect_language_issues(["GREAT PRODUCT!!!", "", "a"])
        >>> print(f"Empty texts: {len(issues['empty'])}")
        >>> print(f"Very short texts: {len(issues['very_short'])}")
        """
        issues = {
            'empty': [],
            'very_short': [],
            'very_long': [],
            'mostly_caps': [],
            'has_urls': [],
            'has_mentions': []
        }
        
        for i, text in enumerate(texts):
            text_str = str(text)
            
            # Check for empty or very short texts
            if pd.isna(text) or text_str.strip() == '':
                issues['empty'].append((i, text))
            elif len(text_str.strip()) < 10:
                issues['very_short'].append((i, text))
            
            # Check for very long texts
            if len(text_str) > 1000:
                issues['very_long'].append((i, text))
            
            # Check for excessive capitalization
            if sum(1 for c in text_str if c.isupper()) > len(text_str) * 0.5:
                issues['mostly_caps'].append((i, text))
            
            # Check for URLs
            if re.search(r'http\S+|www\S+|https\S+', text_str):
                issues['has_urls'].append((i, text))
            
            # Check for social media mentions
            if re.search(r'@\w+|#\w+', text_str):
                issues['has_mentions'].append((i, text))
        
        return issues
    
    def print_issues_summary(self, issues: Dict[str, List[str]]):
        """
        Print a summary of detected text issues.
        
        Parameters:
        -----------
        issues : Dict[str, List[str]]
            Issues dictionary from detect_language_issues()
        """
        print("Text Quality Issues Summary")
        print("="*40)
        
        total_texts = sum(len(issue_list) for issue_list in issues.values())
        
        if total_texts == 0:
            print("✓ No major issues detected in the text data!")
            return
        
        issue_names = {
            'empty': 'Empty texts',
            'very_short': 'Very short texts (< 10 chars)',
            'very_long': 'Very long texts (> 1000 chars)',
            'mostly_caps': 'Texts with excessive capitalization',
            'has_urls': 'Texts containing URLs',
            'has_mentions': 'Texts with @mentions or #hashtags'
        }
        
        for issue_type, issue_list in issues.items():
            if issue_list:
                print(f"⚠ {issue_names[issue_type]}: {len(issue_list)} texts")
        
        print("\nRecommendations:")
        print("- Empty texts: Consider removing these rows")
        print("- Very short texts: May not contain enough information")
        print("- Very long texts: Consider if they're relevant")
        print("- Excessive caps: Usually indicate strong emotion (good for sentiment)")
        print("- URLs/mentions: Usually removed during preprocessing")


def main():
    """
    Example usage of the TextPreprocessor class.
    
    This function demonstrates how to use various preprocessing features.
    """
    print("Text Preprocessing Module - Example Usage")
    print("="*50)
    
    # Example texts (simulating different types of data)
    sample_texts = [
        "This movie was AMAZING!!! 😍 I loved it so much!",
        "Terrible product. Waste of money. 😡 https://example.com",
        "It was okay, nothing special @friend #moviereview",
        "GREAT SERVICE!!! RECOMMEND TO EVERYONE!!!",
        "The food was delicious and the service was excellent.",
        "",
        "Hi",
        "I absolutely love this product! It's the best purchase I've ever made!"
    ]
    
    print("\n1. Basic Text Cleaning:")
    print("-" * 30)
    preprocessor = TextPreprocessor()
    
    for i, text in enumerate(sample_texts[:3], 1):
        print(f"Original {i}: {text}")
        cleaned = preprocessor.basic_clean(text)
        print(f"Cleaned {i}:  {cleaned}\n")
    
    print("\n2. Advanced Text Cleaning:")
    print("-" * 30)
    advanced_preprocessor = TextPreprocessor(
        remove_stopwords=True,
        handle_emojis=True
    )
    
    for i, text in enumerate(sample_texts[:3], 1):
        print(f"Original {i}: {text}")
        cleaned = advanced_preprocessor.advanced_clean(text)
        print(f"Advanced {i}: {cleaned}\n")
    
    print("\n3. Text Statistics:")
    print("-" * 30)
    stats = preprocessor.get_text_statistics(sample_texts)
    print(f"Total texts: {stats['total_texts']}")
    print(f"Average length: {stats['avg_length']:.1f} characters")
    print(f"Most common words: {', '.join([word for word, _ in stats['most_common_words'][:5]])}")
    
    print("\n4. Issue Detection:")
    print("-" * 30)
    issues = preprocessor.detect_language_issues(sample_texts)
    preprocessor.print_issues_summary(issues)
    
    print("\n" + "="*50)
    print("TextPreprocessor is ready to use!")
    print("="*50)


if __name__ == "__main__":
    main()