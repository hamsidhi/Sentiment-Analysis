"""
Setup script for Sentiment Analysis Project
===========================================

This script sets up the sentiment analysis package for easy installation and use.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sentiment-analysis-project",
    version="1.0.0",
    author="AI Assistant",
    author_email="your.email@example.com",
    description="A comprehensive sentiment analysis project for learning and practical use",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sentiment-analysis-project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
    ],
    python_requires=">=3.12",
    install_requires=requirements,
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "pytest>=7.4.0",
        ],
        "optional": [
            "textblob>=0.17.1",
            "emoji>=2.8.0",
            "wordcloud>=1.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sentiment-predict=src.predict:main",
            "sentiment-train=src.train_models:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json"],
    },
)