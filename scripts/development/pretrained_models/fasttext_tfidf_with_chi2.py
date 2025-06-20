# Author: Beatriz Molina Muñiz (GitHub: @Beatriz-MM)
# Last modified: 
# Description: 
# Python version: 3.10.12

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy import sparse
import fasttext
import fasttext.util
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from scipy.sparse import hstack

RANDOM_SEED = 42

# Paths for training datasets
toots_csv_path = '/home/beaunix/TFG/GalMisoCorpus2023/corpus/toots.csv'
tweets_csv_path = '/home/beaunix/TFG/tweets.csv'

# Path to the folder containing all test CSVs
csv_folder = '/home/beaunix/TFG/langdetect/PRUEBA/MiEntreno/CSV_DATA/'
csv_pattern = "csv_gl_comments_*.csv"  # pattern to match all CSVs

# Output path for predictions
output_path = '/home/beaunix/TFG/langdetect/PRUEBA/EntrenoPrevios/Resultados/resultado_predictions_ultima_prueba.csv'


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
        return ' '.join(str(val) for val in sentence_embedding)
        
    except Exception as e:
        print(f"Error generating embeddings for tweet '{tweet}': {e}")
        raise

def load_datasets():
    # Load class 0: non-misogynistic toots
    df_toots = pd.read_csv(toots_csv_path)
    df_toots['content'] = df_toots['content'].apply(preprocess_tweet)
    df_toots = df_toots.dropna(subset=['content']) 
    X_0 = df_toots['content']
    y_0 = pd.Series([0] * len(X_0))

    # Load class 1: misogynistic tweets
    df_tweets = pd.read_csv(tweets_csv_path)
    df_tweets['content'] = df_tweets['content'].apply(preprocess_tweet)
    df_tweets = df_tweets.dropna(subset=['content']) 
    X_1 = df_tweets['content'] 
    y_1 = pd.Series([1] * len(X_1))

    # Merge datasets
    X = pd.concat([X_0, X_1], ignore_index=True)
    y = pd.concat([y_0, y_1], ignore_index=True)

    return X, y

def prepare_embeddings(text_series, fasttext_model):
    sentence_embeddings = text_series.apply(lambda tweet: generate_sentence_embeddings(tweet, fasttext_model))
    sentence_embeddings = np.array(sentence_embeddings.tolist())
    return sentence_embeddings

def train_model(X_train, y_train, combined_features, param_grid):
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

# TF-IDF + chi2 feature selection
vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer()
bow_features = vectorizer.fit_transform(X)
tfidf_features = tfidf_transformer.fit_transform(bow_features)

k_best_selector = SelectKBest(score_func=chi2) 
selected_features = k_best_selector.fit_transform(tfidf_features, y)

# Combine embeddings with TF-IDF
sentence_embeddings = [np.fromstring(embedding, sep=' ') for embedding in sentence_embeddings]
sentence_embeddings = np.array(sentence_embeddings, dtype=np.float32)

print("FORMA setence_embeddings: ", sentence_embeddings.shape)

sparse_embeddings = sparse.csr_matrix(sentence_embeddings)
combined_features = hstack([sparse_embeddings, selected_features])

param_grid = {
    'kernel': ['poly'],
    'C': [1]
}

print("Forma de combined_features:", combined_features.shape)
print("FORMA y:", y.shape)

# Dividimos el dataset en conjuntos de entrenamiento y testing, siendo el 70% entrenamiento y 30% pruebas
X_train, X_test, y_train, y_test = train_test_split(combined_features, y, test_size=0.3, random_state=RANDOM_SEED)


svc = SVC(random_state=RANDOM_SEED)
trained_model = train_model(X_train, y_train, svc, param_grid)

# Generamos la matriz de confusión
y_pred = trained_model.predict(X_test)
cmatrix = confusion_matrix(y_test, y_pred)
cmDisplay = ConfusionMatrixDisplay(cmatrix)

y_pred = trained_model.predict(X_test)
print("Evaluation on Test Set:")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
#cmDisplay.plot()
#plt.show()

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cmatrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Matriz de Confusión")
plt.ylabel("Etiqueta Real")
plt.xlabel("Etiqueta Predicha")
plt.savefig("./Resultados/matriz_confusion_ultima_prueba.png")
plt.close()


# ------------------ PREDICTIONS ON MULTIPLE CSVs ------------------

# Load new data and preprocess
csv_files = glob.glob(os.path.join(csv_folder, csv_pattern))
df_list = []

for file in csv_files:
    try:
        df_temp = pd.read_csv(file)
        df_temp['text'] = df_temp['text'].apply(preprocess_tweet)
        df_temp = df_temp.dropna(subset=['text'])
        df_list.append(df_temp)
        print(f"{os.path.basename(file)} loaded successfully.")
    except Exception as e:
        print(f"Error loading {file}: {e}")

df = pd.concat(df_list, ignore_index=True)

# Generate sentence embeddings for new texts
new_text_embeddings = prepare_embeddings(df['text'], fasttext_model)
new_text_embeddings = [np.fromstring(embedding, sep=' ') for embedding in new_text_embeddings]
new_text_embeddings = np.array(new_text_embeddings, dtype=np.float32)

# Generate BoW + TF-IDF for new texts
new_bow_features = vectorizer.transform(df['text'])
new_tfidf_features = tfidf_transformer.transform(new_bow_features)
new_selected_features = k_best_selector.transform(new_tfidf_features)

# Combine embeddings with TF-IDF features
new_text_embeddings_sparse = sparse.csr_matrix(new_text_embeddings)
new_combined_features = hstack([new_text_embeddings_sparse, new_selected_features])

# Check feature dimensions before prediction
print("Forma de new_combined_features:", new_combined_features.shape)
print("Modelo espera:", trained_model.n_features_in_)

# Predict and save results
predictions = trained_model.predict(new_combined_features)
df['predictions'] = predictions
df[['text', 'predictions']].to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")
