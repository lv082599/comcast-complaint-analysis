########################################################
## Lukas Vitzthum 21-01-25 Projekt Data Analysis v1.3 ##
########################################################

# Importieren der benötigten Bibliotheken
import os
import pandas as pd
import re
import nltk
import joblib
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import save_npz, load_npz
from gensim.models import CoherenceModel, LdaModel
from gensim.corpora import Dictionary

# Sicherstellen, dass NLTK-Ressourcen vorhanden sind
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Datei-Pfad definieren
file_path = 'Comcast.csv'

# Laden des Datensatzes mit Fehlerbehandlung
try:
    df = pd.read_csv(file_path)
    print("Erste Zeilen des Datensatzes:")
    print(df.head(5))
except FileNotFoundError:
    print(f"Fehler: Die Datei '{file_path}' wurde nicht gefunden. Bitte überprüfen Sie den Pfad.")
    exit()

# Prüfen, ob die Spalte Customer Complaint existiert
if 'Customer Complaint' not in df.columns:
    print("Fehler: Die Spalte 'Customer Complaint' existiert nicht im Datensatz.")
    exit()

# Initialisieren von Stopwörtern und Lemmatisierer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Funktion zur parallelen Textbereinigung
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return words

# Textverarbeitung mit paralleler Verarbeitung
df['tokens'] = Parallel(n_jobs=-1)(delayed(preprocess_text)(text) for text in df['Customer Complaint'])
df['cleaned_complaints'] = df['tokens'].apply(lambda x: ' '.join(x))

# Export-Funktion für bereinigte Daten
def export_cleaned_data(dataframe, output_path="bereinigte_daten.csv"):
    if input("Möchten Sie die bereinigten Daten exportieren? (ja/nein): ").strip().lower() == 'ja':
        dataframe[['Customer Complaint', 'cleaned_complaints', 'tokens']].to_csv(output_path, index=False)
        print(f"Bereinigte Daten wurden nach '{output_path}' exportiert.")

export_cleaned_data(df)

# N-Gram Analyse mit TF-IDF
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=1000, min_df=5, max_df=0.8)

# TF-IDF-Matrix speichern/laden
tfidf_path_npz = "tfidf_matrix.npz"
if os.path.exists(tfidf_path_npz):
    tfidf_matrix = load_npz(tfidf_path_npz)
    print("TF-IDF-Matrix geladen aus NPZ-Datei.")
else:
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_complaints'])
    save_npz(tfidf_path_npz, tfidf_matrix)
    print("TF-IDF-Matrix als NPZ gespeichert.")

# Coherence Score für LDA bestimmen
texts = df['tokens'].tolist()
id2word = Dictionary(texts)
corpus = [id2word.doc2bow(text) for text in texts]

lda_path = "lda_model.gensim"
if os.path.exists(lda_path):
    lda_model = LdaModel.load(lda_path)
    print("LDA-Modell geladen aus Datei.")
else:
    best_topic_num = max(
        [(num, CoherenceModel(model=LdaModel(corpus=corpus, id2word=id2word, num_topics=num), 
                             texts=texts, dictionary=id2word, coherence='c_v').get_coherence()) 
         for num in range(2, 15)],
        key=lambda x: x[1]
    )[0]

    lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=best_topic_num)
    lda_model.save(lda_path)
    print(f"LDA-Modell mit {best_topic_num} Themen gespeichert.")

# Elbow Methode für KMeans Clustering
def find_best_k(data, max_k=10):
    distortions = []
    for k in range(2, max_k):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
    
    plt.plot(range(2, max_k), distortions, marker='o')
    plt.xlabel("Anzahl der Cluster (k)")
    plt.ylabel("Distortion (Trägheit)")
    plt.title("Elbow Method zur Bestimmung der optimalen Cluster-Anzahl")
    plt.show()

find_best_k(tfidf_matrix)

# Beste Cluster-Anzahl für KMeans finden
best_k = max(
    [(k, silhouette_score(tfidf_matrix, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(tfidf_matrix)))
     for k in range(2, 10)],
    key=lambda x: x[1]
)[0]

# KMeans-Clustering mit optimaler Cluster-Anzahl
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(tfidf_matrix)

# Cluster-Verteilung visualisieren
sns.countplot(x=df['cluster'])
plt.title('Verteilung der Beschwerden pro Cluster')
plt.xlabel('Cluster')
plt.ylabel('Anzahl der Beschwerden')
plt.show()

# Truncated SVD zur besseren Cluster-Visualisierung
svd = TruncatedSVD(n_components=2, random_state=42)
reduced_data = svd.fit_transform(tfidf_matrix)

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=df['cluster'], cmap='viridis')
plt.title("Truncated SVD-Visualisierung der Cluster")
plt.xlabel("Komponente 1")
plt.ylabel("Komponente 2")
plt.show()

# Funktion zur Erstellung von Wordclouds
def plot_wordcloud(text, title="Wordcloud"):
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(text)
    plt.figure(figsize=(12,6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.show()

# Wordcloud für jede Cluster-Gruppe anzeigen
for cluster_num in range(best_k):
    cluster_text = ' '.join(df[df['cluster'] == cluster_num]['cleaned_complaints'])
    plot_wordcloud(cluster_text, title=f"Wordcloud für Cluster {cluster_num}")
