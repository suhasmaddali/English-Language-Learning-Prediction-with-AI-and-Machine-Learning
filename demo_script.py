import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from spellchecker import SpellChecker
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from transformers import TFGPT2Model, GPT2Tokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('ASAP Dataset/Preprocessed_df.csv')

# Removing any missing values from the data
df = df.dropna(axis = 1, how = 'any')

drop_columns = ['essay_id', 'pos_ratios', 'essay', 'rater1_domain1', 'rater2_domain1']
df.drop(drop_columns, axis = 1, inplace = True)
df = df[df.essay_set == 1]
df_sample = df.sample(n = 2)

def dataset_preparation(data, target = 'domain1_score'):
    
    X = data.drop([target], axis = 1)
    y = data[target]
    
    return X, y

X, y = dataset_preparation(df_sample)

BATCH_SIZE = 16
MAX_LENGTH = 512

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
gpt_model = TFGPT2Model.from_pretrained('gpt2')

train_encodings = tokenizer(list(X['preprocessed_text']), truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors='tf')
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y)).batch(BATCH_SIZE)

sample_embeddings = []
for batch in tqdm(train_dataset):
    sample_embeddings.append(gpt_model(batch[0]['input_ids'])[0][:, -1, :])
sample_embeddings = tf.concat(sample_embeddings, axis=0)

def preprocessing_function(X_train):

    with open('Models/bow_vectorizer1.pkl', 'rb') as f:
        bow_vectorizer = pickle.load(f)
    X_train_bow = bow_vectorizer.transform(X_train['preprocessed_text']).toarray()

    with open('Models/tfidf_vectorizer1.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    X_train_tfidf = tfidf_vectorizer.transform(X_train['preprocessed_text']).toarray()

    with open('Models/scaler1.pkl', 'rb') as f:
        scaler = pickle.load(f)
    X_train_bow = scaler.transform(X_train_bow)
    X_train_tfidf = scaler.transform(X_train_tfidf)

    pca_bow = PCA()
    with open('Models/pca1.pkl', 'rb') as f:
        pca_bow = pickle.load(f)
    X_train_bow_pca = pca_bow.transform(X_train_bow)

    variance_ratio_bow = np.cumsum(pca_bow.explained_variance_ratio_)
    n_components_bow = np.argmax(variance_ratio_bow >= 0.95) + 1
    X_train_bow_pca = X_train_bow_pca[:, :n_components_bow]

    with open('Models/pca2.pkl', 'rb') as f:
        pca_tfidf = pickle.load(f)
    X_train_tfidf_pca = pca_tfidf.transform(X_train_tfidf)

    variance_ratio_tfidf = np.cumsum(pca_tfidf.explained_variance_ratio_)
    n_components_tfidf = np.argmax(variance_ratio_tfidf >= 0.95) + 1
    X_train_tfidf_pca = X_train_tfidf_pca[:, :n_components_tfidf]

    X_train_bow = tf.convert_to_tensor(X_train_bow_pca, dtype = tf.float32)

    X_train_tfidf = tf.convert_to_tensor(X_train_tfidf_pca, dtype = tf.float32)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.drop(['preprocessed_text'], axis = 1))
    X_train_features = tf.constant(X_train_scaled.astype('float32'))

    return X_train_features, X_train_bow

X_features, X_bow = preprocessing_function(X)
sample_embeddings = tf.concat([sample_embeddings, X_features, X_bow], axis = 1)

with open('Models/best_model.pkl', 'rb') as f:
    model_load = pickle.load(f)

y_predictions = model_load.predict(sample_embeddings)

X['y_predictions'] = y_predictions
X['y_actual'] = y

print(X[['preprocessed_text', 'y_predictions', 'y_actual']].head())
