########################################################
## Lukas Vitzthum 18-02-25 Projekt Data Analysis v1.4 ##
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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Herunterladen und Validieren der NLTK-Ressourcen
nltk_resources = ['stopwords', 'punkt', 'wordnet']
for resource in nltk_resources:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource)

# Datei-Pfad
file_path = 'Comcast.csv'

# Laden des Datensatzes mit Fehlerbehandlung
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        df.dropna(subset=['Customer Complaint'], inplace=True)  # Fehlende Werte entfernen
        return df
    except FileNotFoundError:
        print(f"Fehler: Die Datei '{file_path}' wurde nicht gefunden. Bitte überprüfen Sie den Pfad.")
        exit()

df = load_dataset(file_path)

# Funktion zur Textbereinigung mit N-Gram-Integration
def clean_text(text, ngram_range=(1, 2)):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))  # Entfernt Sonderzeichen und Zahlen
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    
    # N-Gram-Feature-Erstellung
    ngrams = ['_'.join(words[i:i+ngram_range[1]]) for i in range(len(words)-ngram_range[1]+1)]
    return ' '.join(words + ngrams)

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(word) for word in word_tokenize(text)]

# Anwenden der Bereinigungsfunktionen mit Fortschrittsanzeige
tqdm.pandas(desc="Bereinige Texte")
df['cleaned_complaints'] = df['Customer Complaint'].progress_apply(lambda x: clean_text(x))
df['tokens'] = df['cleaned_complaints'].progress_apply(lambda x: lemmatize_text(x))
df['num_tokens'] = df['tokens'].apply(len)

def export_cleaned_data(dataframe, output_path="bereinigte_daten.csv"):
    dataframe[['Customer Complaint', 'cleaned_complaints', 'tokens', 'num_tokens']].to_csv(output_path, index=False)
    print(f"Bereinigte Daten wurden nach '{output_path}' exportiert.")

export_cleaned_data(df)

# TF-IDF-Vektorisierung
vectorizer_tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
tfidf_matrix = vectorizer_tfidf.fit_transform(df['cleaned_complaints'])
print(f"TF-IDF-Matrix: {tfidf_matrix.shape}")

# CountVectorizer (Bag-of-Words)
vectorizer_count = CountVectorizer(max_features=1000, ngram_range=(1,2))
count_matrix = vectorizer_count.fit_transform(df['cleaned_complaints'])
print(f"Count-Matrix: {count_matrix.shape}")

# Themenmodellierung mit LDA
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(tfidf_matrix)

# Themenmodellierung mit NMF
nmf = NMF(n_components=5, random_state=42)
nmf.fit(tfidf_matrix)

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

# Export der Ergebnisse als TXT-Datei
with open("summary.txt", "w") as file:
    file.write("Comcast Complaint Analysis\n")
    file.write("==========================\n\n")
    file.write(f"Anzahl der Beschwerden: {len(df)}\n")
    file.write(f"Durchschnittliche Anzahl von Tokens pro Beschwerde: {df['num_tokens'].mean():.2f}\n\n")
    file.write("LDA-Themen:\n")
    for index, topic in enumerate(lda.components_):
        file.write(f"Thema {index}: {[vectorizer_tfidf.get_feature_names_out()[i] for i in topic.argsort()[-10:]]}\n")
    file.write("\nNMF-Themen:\n")
    for index, topic in enumerate(nmf.components_):
        file.write(f"Thema {index}: {[vectorizer_tfidf.get_feature_names_out()[i] for i in topic.argsort()[-10:]]}\n")
    file.write("\nCluster-Verteilung:\n")
    file.write(str(df['cluster'].value_counts()))
    file.write("\nExport abgeschlossen.\n")

print("Zusammenfassung wurde in 'summary.txt' gespeichert.")
