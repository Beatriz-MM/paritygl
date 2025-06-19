# Author: Beatriz Molina Muñiz (GitHub: @Beatriz-MM)
# Last modified: 
# Description:
# Python version: 3.10.12

import os
import glob
import re
import sys
import numpy as np
import pandas as pd
import fasttext
import fasttext.util
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split


# Paths for training datasets
toots_csv_path = '/home/beaunix/TFG/GalMisoCorpus2023/corpus/toots.csv'
tweets_csv_path = '/home/beaunix/TFG/tweets.csv'

# Path to the folder containing all test CSVs
csv_folder = '/home/beaunix/TFG/langdetect/PRUEBA/Prueba_embeddings/'

# Output path for predictions
output_path = '/home/beaunix/TFG/langdetect/PRUEBA/Prueba_embeddings/Ultimas_pruebas/ultimos_resultados/resultado_predictions_sinfunc_matriz.csv'


# ------------------ PREPROCESSING ------------------

def preprocess_tweet(tweet):
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
    try: 
        tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
        tokens = tokenizer.tokenize(tweet)

        if not tokens:
            raise ValueError(f"No tokens found for tweet '{tweet}'")

        embeddings = [fasttext_model.get_word_vector(word) for word in tokens]
        sentence_embedding = sum(embeddings) / len(embeddings)
        return sentence_embedding
    
    except Exception as e:
        print(f"Error generating embeddings for tweet '{tweet}': {e}")
        raise

def load_datasets():

    # Dataset for class 0 (non-misogynistic toots)
    df_toots = pd.read_csv(toots_csv_path)
    df_toots['content'] = df_toots['content'].apply(preprocess_tweet)
    df_toots = df_toots.dropna(subset=['content'])  # Remove rows with empty toots
    X_0 = df_toots['content']
    y_0 = pd.Series([0] * len(X_0))

    # Dataset for class 1 (misogynistic tweets)
    df_tweets = pd.read_csv(tweets_csv_path)
    df_tweets['content'] = df_tweets['content'].apply(preprocess_tweet)
    df_tweets = df_tweets.dropna(subset=['content'])  # Remove rows with empty tweets
    X_1 = df_tweets['content'] 
    y_1 = pd.Series([1] * len(X_1))

    # Combine both datasets
    X = pd.concat([X_0, X_1], ignore_index=True)
    y = pd.concat([y_0, y_1], ignore_index=True)

    return X, y

def prepare_embeddings(text_series, fasttext_model):
    sentence_embeddings = text_series.apply(lambda tweet: generate_sentence_embeddings(tweet, fasttext_model))
    sentence_embeddings = np.array(sentence_embeddings.tolist())
    return sentence_embeddings

def train_model(X_train, y_train, param_grid):
    grid_search = GridSearchCV(estimator=SVC(), param_grid=param_grid, scoring='f1', cv=10)
    grid_search.fit(X_train, y_train)
    trained_model = grid_search.best_estimator_
    return trained_model


# ------------------ MODEL TRAINING ------------------

# Download and load FastText Galician model
if not os.path.exists('cc.gl.300.bin'):
    fasttext.util.download_model('gl', if_exists='ignore')
fasttext_model = fasttext.load_model('cc.gl.300.bin')


# Load and preprocess the training data
X, y = load_datasets()
sentence_embeddings = prepare_embeddings(X, fasttext_model)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(sentence_embeddings, y, test_size=0.3)

# Define parameters for the SVM model
param_grid = {
    'kernel': ['poly'],
    'C': [1]
}

# Train the SVM model
trained_model = train_model(X_train, y_train, param_grid) 

# Evaluate the model on test data
y_pred = trained_model.predict(X_test)
print("Evaluation on the test set:")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Check if y_pred changes
y_pred_before = y_pred.copy()

# Plot the confusion matrix
cmatrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cmatrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig("./ultimos_resultados/matriz_confusion_sinfunc.png")
plt.show()
plt.close()

# Compare y_pred after plot
assert np.array_equal(y_pred, y_pred_before), "y_pred has been modified!"


# ------------------ PREDICTIONS ON MULTIPLE CSVs ------------------

csv_files = glob.glob(os.path.join(csv_folder, 'csv_gal_comments_*.csv'))
df_list = []

# Load and preprocess each CSV
for file in csv_files:
    try:
        df = pd.read_csv(file)
        df['text'] = df['text'].apply(preprocess_tweet)
        df = df.dropna(subset=['text'])
        df_list.append(df)
        print(f"{os.path.basename(file)} loaded successfully.")
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Combine all test data
df_all = pd.concat(df_list, ignore_index=True)

# Generate embeddings and predict
new_text_embeddings = prepare_embeddings(df_all['text'], fasttext_model)
predictions = trained_model.predict(new_text_embeddings)
df_all['predictions'] = predictions

# Save predictions to output CSV
df_all[['text', 'predictions']].to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")

# Print predictions
print("Predictions on the combined corpus:")
print(df_all[['text', 'predictions']])