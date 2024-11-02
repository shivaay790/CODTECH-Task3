# Sentiment Analysis on Amazon Reviews

This project implements a comprehensive sentiment analysis model aimed at classifying Amazon product reviews into three categories: positive, neutral, and negative. By leveraging natural language processing (NLP) techniques, word embeddings via Word2Vec, and machine learning algorithms, the project aims to derive insights from textual data that can be invaluable for businesses and consumers alike.

## Table of Contents

- [Sentiment Analysis on Amazon Reviews](#sentiment-analysis-on-amazon-reviews)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
    - [Key Components](#key-components)
  - [Dataset](#dataset)
    - [Data Source](#data-source)
  - [Requirements](#requirements)
    - [Installation of Dependencies](#installation-of-dependencies)
    - [Data Preprocessing](#data-preprocessing)
    - [Feature Extraction](#feature-extraction)
    - [Model Training](#model-training)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
    - [Model Evaluation](#model-evaluation)

## Project Overview

The primary objective of this project is to classify the sentiment of product reviews using a combination of text data and numerical features. The project employs multiple classifiers, including Logistic Regression and Support Vector Classifier (SVC), evaluating their effectiveness in sentiment classification. 

### Key Components

1. **Data Cleaning**: Handling missing or inconsistent data entries.
2. **Text Preprocessing**: Transforming raw text into a clean and usable format.
3. **Feature Engineering**: Generating numerical representations of text data.
4. **Model Selection**: Training and evaluating various machine learning models.
5. **Performance Evaluation**: Measuring the accuracy and efficiency of the models.

## Dataset

The dataset for this project is sourced from Amazon reviews. The key columns in the dataset include:

- **`review_title`**: The title of the customer review.
- **`review_content`**: The full text of the customer review.
- **`about_product`**: Brief information about the product.
- **`discounted_price`**: The price after discounts.
- **`discount_percentage`**: The discount applied on the product.
- **`rating`**: The rating given by the customer (on a scale of 1 to 5).
- **`rating_count`**: Total number of ratings received.
- **`actual_price`**: The original price of the product.

### Data Source

The dataset can be accessed via [Kaggle](https://www.kaggle.com/) (provide a direct link if available). 

## Requirements

The following Python libraries are required to run the project:

- `numpy`: For numerical operations.
- `pandas`: For data manipulation and analysis.
- `matplotlib`: For plotting and visualizing data.
- `seaborn`: For statistical data visualization.
- `nltk`: For natural language processing tasks.
- `gensim`: For working with word embeddings.
- `scikit-learn`: For implementing machine learning algorithms.

### Installation of Dependencies

You can install the required libraries using:

```bash
pip install numpy pandas matplotlib seaborn nltk gensim scikit-learn
```

### Data Preprocessing

The data preprocessing phase involves several steps to clean and prepare the text data for analysis:

1) Handling Missing Values: Identifying and addressing missing or invalid entries in the dataset.
2) Text Cleaning: Removing special characters, HTML tags, and irrelevant information.
3) Lowercasing: Converting all text to lowercase to ensure uniformity.
4) Tokenization: Splitting text into individual words (tokens).
5) Lemmatization: Reducing words to their base form to ensure similar meanings are treated equally.
6) Stopword Removal: Filtering out common words (e.g., "and", "the") that do not contribute to sentiment.

### Feature Extraction

Two methods of feature extraction are employed:

1) TF-IDF Vectorization: Converts the cleaned text data into numerical format using the Term Frequency-Inverse Document Frequency method.
2) Word2Vec: Creates word embeddings that represent words as vectors in a continuous vector space.

### Model Training 

The project implements the following classifiers:

1) Logistic Regression: A baseline classifier for sentiment classification.
2) Support Vector Classifier (SVC): A more advanced model that can handle non-linear data.

### Hyperparameter Tuning

Grid Search is utilized for hyperparameter optimization to improve model performance. Parameters for both classifiers are specified, and the best-performing configurations are selected.

```python
grid_search_lr = GridSearchCV(estimator=LR_model2, param_grid=lr_params, scoring='accuracy', cv=5)
grid_search_svc = GridSearchCV(estimator=svc_model2, param_grid=svc_params, scoring='accuracy', cv=8)
```

### Model Evaluation

The performance of each model is evaluated using:

1) Accuracy: The percentage of correct predictions.
2) Classification Report: Provides precision, recall, and F1-score for each class.
3) Confusion Matrix: Visual representation of true vs. predicted classifications.