# Author: Beatriz Molina Muñiz (GitHub: @Beatriz-MM)
# Last modified: 23/06/2025
# Description: Loads and preprocesses two labeled datasets (non-misogynistic toots and misogynistic tweets), 
# generates sentence embeddings using a FastText Galician model, combines them with TF-IDF features (without chi2 selection) 
# and trains an SVM classifier to detect misogynistic content.
# The model is evaluated on a separate sample dataset of Instagram comments and used to generate predictions.
# Python version: 3.10.12

import os
import re
import numpy as np
import pandas as pd
import fasttext
import fasttext.util
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from scipy import sparse
from scipy.sparse import hstack

RANDOM_SEED = 42

# Paths for training datasets
toots_csv_path = ""# Example: "~/corpus/toots.csv"
tweets_csv_path = ""# Example: "~/corpus/tweets.csv"

# Path sample CSV
csv_sample = ""# Example: "~/petrained_models/csv_gl_comments_sample.csv"

# Output path for predictions and matrix
output_path = ""# Example: "~/Results_without_chi2/predictions_without_chi2.csv"
confusion_matrix_path = ""# Example: "~/Results_without_chi2/confusion_matrix_without_chi2.png"


# ------------------ PREPROCESSING ------------------

def preprocess_tweet(tweet):
    """
    Clean and normalize a tweet by removing noise and standardizing the text.

    Args:
        tweet (str): Raw tweet text.
        
    Returns:
        str or None: Cleaned tweet text, or None if input is invalid or results in empty string.
    """
    if not isinstance(tweet, str) or tweet is None:
        return None

    tweet = tweet.lower()
    tweet = re.sub(r'\n', '', tweet)
    tweet = re.sub(r'http://t.co/[a-zA-Z0-9]+', 'http://t.co', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'::', '', tweet)
    tweet = re.sub(r'#(\w+)', r'\1', tweet)
    tweet = re.sub(r'(.)\1{2,}', r'\1', tweet)
    
    if not tweet.strip():
        return None
    return tweet

def generate_sentence_embeddings(tweet, fasttext_model):
    """
    Generate a sentence embedding for a given tweet by averaging FastText word vectors.

    Args:
        tweet (str): Cleaned input text.
        fasttext_model: A preloaded FastText model.

    Returns:
        str: Space-separated string of the averaged word embedding vector.

    Raises:
        ValueError: If no tokens are found.
        Exception: If embedding generation fails unexpectedly.
    """
    try:
        tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
        tokens = tokenizer.tokenize(tweet)
        if not tokens:
            raise ValueError(f"No tokens found for tweet '{tweet}'")

        embeddings = [fasttext_model.get_word_vector(word) for word in tokens]
        sentence_embedding = sum(embeddings) / len(embeddings)
        return ' '.join(str(val) for val in sentence_embedding)

    except Exception as e:
        print(f"Error generating embeddings for tweet '{tweet}': {e}")
        raise


def load_datasets():
    """
    Load, clean, and label two datasets: non-misogynistic toots (class 0) and misogynistic tweets (class 1).

    Returns:
        tuple: 
            - X (pandas.Series): Combined and preprocessed text data.
            - y (pandas.Series): Corresponding binary labels (0 for non-misogynistic, 1 for misogynistic).
    """
    df_toots = pd.read_csv(toots_csv_path)
    df_toots['content'] = df_toots['content'].apply(preprocess_tweet)
    df_toots = df_toots.dropna(subset=['content'])
    X_0 = df_toots['content']
    y_0 = pd.Series([0] * len(X_0))

    df_tweets = pd.read_csv(tweets_csv_path)
    df_tweets['content'] = df_tweets['content'].apply(preprocess_tweet)
    df_tweets = df_tweets.dropna(subset=['content'])
    X_1 = df_tweets['content']
    y_1 = pd.Series([1] * len(X_1))

    X = pd.concat([X_0, X_1], ignore_index=True)
    y = pd.concat([y_0, y_1], ignore_index=True)

    return X, y


def prepare_embeddings(text_series, fasttext_model):
    """
    Generate sentence embeddings for a series of texts using a FastText model.

    Args:
        text_series (pandas.Series): Series of cleaned text strings.
        fasttext_model: A preloaded FastText model.

    Returns:
        numpy.ndarray: Array of sentence embeddings as space-separated strings.
    """
    sentence_embeddings = text_series.apply(lambda tweet: generate_sentence_embeddings(tweet, fasttext_model))
    sentence_embeddings = np.array(sentence_embeddings.tolist())
    return sentence_embeddings


def train_model(X_train, y_train, combined_features, param_grid):
    """
    Train an SVM model with hyperparameter tuning using grid search.

    Args:
        X_train (array-like or sparse matrix): Training features.
        y_train (array-like): Training labels.
        combined_features (sklearn.svm.SVC): Base model to train.
        param_grid (dict): Grid of hyperparameters to search.

    Returns:
        sklearn.svm.SVC: Best estimator found during grid search.
    """
    grid_search = GridSearchCV(estimator=combined_features, param_grid=param_grid, scoring='f1', cv=10)
    grid_search.fit(X_train, y_train)
    trained_model = grid_search.best_estimator_
    return trained_model


# ------------------ MODEL TRAINING ------------------

# Download and load FastText Galician model
if not os.path.exists('cc.gl.300.bin'):
    fasttext.util.download_model('gl', if_exists='ignore')
fasttext_model = fasttext.load_model('cc.gl.300.bin')

# Prepare data and generate embeddings
X, y = load_datasets()
sentence_embeddings = prepare_embeddings(X, fasttext_model)

# TF-IDF representation
vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer()
bow_features = vectorizer.fit_transform(X)
selected_features = tfidf_transformer.fit_transform(bow_features)

# Combine embeddings with TF-IDF
sentence_embeddings = [np.fromstring(embedding, sep=' ') for embedding in sentence_embeddings]
sentence_embeddings = np.array(sentence_embeddings, dtype=np.float32)
print("Shape of sentence_embeddings:", sentence_embeddings.shape)

sparse_embeddings = sparse.csr_matrix(sentence_embeddings)
combined_features = hstack([sparse_embeddings, selected_features])

# Parameter grid for SVM
param_grid = {
    'kernel': ['poly'],
    'C': [1]
}

print("Shape of combined_features:", combined_features.shape)
print("Shape of y:", y.shape)

# Split the dataset into training and test sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(combined_features, y, test_size=0.3, random_state=RANDOM_SEED)


svc = SVC(random_state=RANDOM_SEED)
print("Starting Grid Search...")
trained_model = train_model(X_train, y_train, svc, param_grid)

# MODEL EVALUATION
y_pred = trained_model.predict(X_test)
cmatrix = confusion_matrix(y_test, y_pred)

print("Evaluation on Test Set:")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Save confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(cmatrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig(confusion_matrix_path)
plt.close()


# ------------------ PREDICTIONS ON MULTIPLE CSVs ------------------

# Load new data and preprocess
try:
    df_all = pd.read_csv(csv_sample)
    df_all['text'] = df_all['text'].apply(preprocess_tweet)
    df_all = df_all.dropna(subset=['text'])
    print(f"{os.path.basename(csv_sample)} loaded and preprocessed successfully.")
except Exception as e:
    print(f"Error loading {csv_sample}: {e}")
    df_all = pd.DataFrame()


# Generate sentence embeddings for new texts
new_text_embeddings = prepare_embeddings(df_all['text'], fasttext_model)
new_text_embeddings = [np.fromstring(embedding, sep=' ') for embedding in new_text_embeddings]
new_text_embeddings = np.array(new_text_embeddings, dtype=np.float32)

# Generate BoW + TF-IDF for new texts
new_bow_features = vectorizer.transform(df_all['text'])
new_selected_features = tfidf_transformer.transform(new_bow_features)

# Combine embeddings with TF-IDF features
new_text_embeddings_sparse = sparse.csr_matrix(new_text_embeddings)
new_combined_features = hstack([new_text_embeddings_sparse, new_selected_features])

# Check feature dimensions before prediction
print("Shape of new_combined_features:", new_combined_features.shape)
print("Model expects:", trained_model.n_features_in_)

# Predict and save results
predictions = trained_model.predict(new_combined_features)
df_all['predictions'] = predictions
df_all[['text', 'predictions']].to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")
