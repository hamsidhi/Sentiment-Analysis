import pandas as pd
import os
from pathlib import Path

def find_text_column(df):
    """Smartly find text/review column"""
    text_keywords = ['text', 'review', 'comment', 'message', 'content', 'reviewtext', 'tweet']
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword.lower() in col_lower for keyword in text_keywords):
            return col
    return df.columns[0]

def find_sentiment_column(df):
    """Smartly find sentiment/rating column"""
    sentiment_keywords = ['sentiment', 'label', 'score', 'overall', 'rating', 'target', 'polarity']
    for col in df.columns:
        if any(keyword.lower() in col.lower() for keyword in sentiment_keywords):
            return col
    return None

def load_universal_csv(file_path):
    """Universal CSV loader - handles ALL encodings without chardet"""
    print(f"📂 Loading: {os.path.basename(file_path)}")
    
    # Try multiple encodings (NO chardet needed)
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
            print(f"✅ Loaded with {encoding}")
            break
        except (UnicodeDecodeError, UnicodeError):
            print(f"⏭️  Skipping {encoding}")
            continue
    
    if df is None:
        raise ValueError(f"❌ Could not read {file_path} with any encoding")
    
    print(f"📊 {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"📋 Columns: {list(df.columns)}")
    
    text_col = find_text_column(df)
    sentiment_col = find_sentiment_column(df)
    
    print(f"🔍 Text: '{text_col}' | Sentiment: '{sentiment_col}'")
    
    # Clean data
    if sentiment_col and sentiment_col in df.columns:
        df_clean = df[[text_col, sentiment_col]].dropna()
        df_clean.columns = ['text', 'sentiment']
    else:
        df_clean = df[[text_col]].dropna()
        df_clean.columns = ['text']
    
    print(f"✅ Cleaned: {df_clean.shape[0]} rows")
    if 'sentiment' in df_clean:
        print(f"📈 Sentiments:\n{df_clean['sentiment'].value_counts().head()}")
    
    return df_clean

# MAIN
if __name__ == "__main__":
    folder = "data/raw/"
    Path("data/processed/").mkdir(parents=True, exist_ok=True)
    
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    all_data = []
    
    print(f"🔍 Found {len(files)} CSV files")
    
    for file in files:
        print(f"\n{'='*60}")
        print(f"📖 {file}")
        filepath = os.path.join(folder, file)
        
        try:
            df = load_universal_csv(filepath)
            all_data.append(df)
            print("✅ SUCCESS")
        except Exception as e:
            print(f"💥 FAILED: {e}")
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        print(f"\n🎉 COMBINED: {final_df.shape[0]} TOTAL REVIEWS")
        
        if 'sentiment' in final_df.columns:
            print("📊 FINAL COUNTS:")
            print(final_df['sentiment'].value_counts())
        
        final_df.to_csv("data/processed/sentiment_data.csv", index=False, encoding='utf-8')
        print("💾 SAVED: data/processed/sentiment_data.csv")
    else:
        print("❌ NO DATA PROCESSED")
