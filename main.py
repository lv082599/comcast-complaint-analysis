########################################################
## Lukas Vitzthum 17-02-25 Projekt Data Analysis v1.3 ##
########################################################

# Importieren der benötigten Bibliotheken
import os
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from multiprocessing import cpu_count

# Herunterladen und Validieren der NLTK-Ressourcen
nltk_resources = ['stopwords', 'punkt', 'wordnet']
for resource in nltk_resources:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource)

# Datei-Pfad
df_path = 'Comcast.csv'

def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        df.dropna(subset=['Customer Complaint'], inplace=True)
        return df
    except FileNotFoundError:
        print(f"Fehler: Die Datei '{file_path}' wurde nicht gefunden. Bitte überprüfen Sie den Pfad.")
        exit()

df = load_dataset(df_path)

# Funktion zur Textbereinigung mit Multi-Processing
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(word) for word in word_tokenize(text)]

# Bereinigung parallelisieren
df['cleaned_complaints'] = Parallel(n_jobs=cpu_count())(delayed(clean_text)(text) for text in df['Customer Complaint'])
df['tokens'] = Parallel(n_jobs=cpu_count())(delayed(lemmatize_text)(text) for text in df['cleaned_complaints'])
df['num_tokens'] = df['tokens'].apply(len)

def export_cleaned_data(dataframe, output_path="bereinigte_daten.csv"):
    dataframe[['Customer Complaint', 'cleaned_complaints', 'tokens', 'num_tokens']].to_csv(output_path, index=False)
    print(f"Bereinigte Daten wurden nach '{output_path}' exportiert.")

export_cleaned_data(df)

# TF-IDF-Vektorisierung
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
tfidf_matrix = vectorizer.fit_transform(df['cleaned_complaints'])

# Themenmodellierung mit LDA
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(tfidf_matrix)

# Themenmodellierung mit NMF
nmf = NMF(n_components=5, random_state=42)
nmf.fit(tfidf_matrix)

# Coherence Score Evaluation mit Multi-Threading
from sklearn.metrics.pairwise import cosine_similarity

def calculate_coherence(model, matrix):
    topics = model.components_
    topic_words = np.argsort(topics, axis=1)[:, -10:]
    similarities = Parallel(n_jobs=cpu_count())(delayed(lambda words: cosine_similarity(matrix[:, words].T).mean())(words) for words in topic_words)
    return np.mean(similarities)

lda_coherence = calculate_coherence(lda, tfidf_matrix)
nmf_coherence = calculate_coherence(nmf, tfidf_matrix)
print(f"LDA Coherence Score: {lda_coherence:.4f}")
print(f"NMF Coherence Score: {nmf_coherence:.4f}")

# Clustering mit K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(tfidf_matrix)
df['cluster'] = kmeans.labels_

# PCA für Visualisierung
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(tfidf_matrix.toarray())
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=df['cluster'], cmap='viridis')
plt.title("PCA-Visualisierung der Cluster")
plt.xlabel("Komponente 1")
plt.ylabel("Komponente 2")
plt.show()

# Wordcloud erstellen
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=200, random_state=42).generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Wordcloud der häufigsten Wörter in Comcast Beschwerden", fontsize=16, fontweight='bold')
    plt.show()

generate_wordcloud(' '.join(df['cleaned_complaints']))

# Ergebnisse exportieren
def export_results():
    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write("Comcast Complaint Analysis Summary\n")
        f.write("===============================\n\n")
        f.write(f"LDA Coherence Score: {lda_coherence:.4f}\n")
        f.write(f"NMF Coherence Score: {nmf_coherence:.4f}\n\n")
        f.write("Cluster-Verteilung:\n")
        f.write(df['cluster'].value_counts().to_string())
        f.write("\n\nThemenmodellierung:\n")
        for index, topic in enumerate(lda.components_):
            f.write(f"Thema {index}: {', '.join([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])}\n")
        print("Ergebnisse wurden in 'summary.txt' gespeichert.")

export_results()
