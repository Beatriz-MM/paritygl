# Author: Beatriz Molina Muñiz (GitHub: @Beatriz-MM)
# Last modified: 05/06/2025
# Description: This script trains a linear SVM model to classify Instagram comments as negative (label=1) or not (label=0).  
# It combines a set of confirmed negative examples with a dataset, uses FastText embeddings (cc.gl.300.bin), 
# and evaluates the model performance.
# Python version: 3.10.12

import os
import pandas as pd
import numpy as np
import joblib
from scipy import sparse
import fasttext
import fasttext.util
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import TweetTokenizer
from sklearn.svm import SVC
from nltk.tokenize import TweetTokenizer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack


RANDOM_SEED = 42

# Paths to input datasets
negative_dataset_path = "/home/beaunix/TFG/langdetect/PRUEBA/MiEntreno/negative_dataset.csv" 
csv_path = "/home/beaunix/TFG/langdetect/PRUEBA/MiEntreno/comentarios_etiquetados.csv"

# Paths to output files
report_path = "/home/beaunix/TFG/langdetect/PRUEBA/MiEntreno/classification_comments_report_batches.txt"
matrix_path = "/home/beaunix/TFG/langdetect/PRUEBA/MiEntreno/confusion_matrix_training_batches.png"
output_path = '/home/beaunix/TFG/langdetect/PRUEBA/MiEntreno/result_predictions_training_batches.csv'

# ---------- FUNCTIONS ----------

def load_dataset():
    # Load confirmed negative comments and label them as 1
    neg_df = pd.read_csv(negative_dataset_path, usecols=['text'])
    neg_df['label'] = 1  

    # Load full labeled dataset (with columns: id, language, text, label)
    dataset_df = pd.read_csv(csv_path, usecols=['text', 'label'])

    # Unir ambos datasets
    corpus_df = pd.concat([dataset_df, neg_df], ignore_index=True)

    # Remove duplicate texts
    corpus_df = corpus_df.drop_duplicates(subset='text')

    # Shuffle the entire dataset
    corpus_df = corpus_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print("Label distribution in the combined corpus:")
    print(corpus_df['label'].value_counts())

    return corpus_df

def get_fasttext_model():
    fasttext.util.download_model('gl', if_exists='ignore')
    return fasttext.load_model('cc.gl.300.bin')

def plot_conf_matrix(y_true, y_pred, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f" Confusion Matrix saved at {save_path}")
    plt.show()
    plt.close()

def prepare_embeddings(text_series, fasttext_model, batch_size=1000):
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
    embeddings = []

    for start in range(0, len(text_series), batch_size):
        batch_texts = text_series[start:start + batch_size]
        batch_embeddings = []

        for text in batch_texts:
            tokens = tokenizer.tokenize(text)
            vectors = [fasttext_model.get_word_vector(token) for token in tokens]
            if not vectors:
                batch_embeddings.append(np.zeros(300))
            else:
                batch_embeddings.append(np.mean(vectors, axis=0))

        embeddings.extend(batch_embeddings)

    return np.array(embeddings, dtype=np.float32)


def train_model(X_train, y_train, combined_features, param_grid):
    grid_search = GridSearchCV(estimator=combined_features, param_grid=param_grid, scoring='f1', cv=10)
    grid_search.fit(X_train, y_train)
    trained_model = grid_search.best_estimator_
    return trained_model


# ---------- MAIN ----------

# Load and label data
df = load_dataset()

# Delete empty text o NaN
df = df.dropna(subset=['text'])                      
df = df[df['text'].str.strip() != ""]          

X = df['text']
y = df['label']

# Download and load FastText Galician model
if not os.path.exists('cc.gl.300.bin'):
    fasttext.util.download_model('gl', if_exists='ignore')
fasttext_model = fasttext.load_model('cc.gl.300.bin')

# Embeddings en batches
sentence_embeddings = prepare_embeddings(X, fasttext_model, batch_size=1000)
print("FORMA sentence_embeddings:", sentence_embeddings.shape)

# TF-IDF + chi2 feature selection
vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer()
bow_features = vectorizer.fit_transform(X)
tfidf_features = tfidf_transformer.fit_transform(bow_features)

k_best_selector = SelectKBest(score_func=chi2) 
selected_features = k_best_selector.fit_transform(tfidf_features, y)

# Combine embeddings with TF-IDF features
sparse_embeddings = sparse.csr_matrix(sentence_embeddings)
combined_features = hstack([sparse_embeddings, selected_features])

print("Forma de combined_features:", combined_features.shape)
print("FORMA y:", y.shape)

sparse_embeddings = sparse.csr_matrix(sentence_embeddings)
combined_features = hstack([sparse_embeddings, selected_features])

param_grid = {
    'kernel': ['poly'],
    'C': [1]
}

print("Forma de combined_features:", combined_features.shape)
print("FORMA y:", y.shape)

# Split the dataset into training and test sets, with 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(combined_features, y, test_size=0.3, random_state=RANDOM_SEED)

svc = SVC(random_state=RANDOM_SEED)
trained_model = train_model(X_train, y_train, svc, param_grid)

# Save trained model
joblib.dump(trained_model, 'model_SVM_instagram_fasttext.pkl')


# ----------- EVALUATION -----------

y_pred = trained_model.predict(X_test)

print("\n Evaluation on Test Set: \n")
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)

print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Save classification report to file
report = classification_report(y_test, y_pred)
with open(report_path, "w", encoding="utf-8") as f:
    f.write("Classification Report\n\n")
    f.write(report)

print(f" Classification report saved to {report_path}")

# Confusion Matrix
plot_conf_matrix(y_test, y_pred, "Confusion Matrix", matrix_path)

# Save classification report
X_test_raw = pd.DataFrame({
    'text': X_test,
    'true_label': y_test.values,
    'predicted_label': y_pred
})

X_test_raw.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f" Predictions saved to {output_path}")