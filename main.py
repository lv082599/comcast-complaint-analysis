########################################################
## Lukas Vitzthum 21-01-25 Projekt Data Analysis v1.1 ##
########################################################

# Importieren der benötigten Bibliotheken
import os
import pandas as pd
import re
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Herunterladen der NLTK-Ressourcen
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Definieren des Datei-Pfads
file_path = os.path.join('daten', 'Comcast.csv')

# Laden des Datensatzes mit einer Fehlerbehandlung
try:
    df = pd.read_csv(file_path)
    print("Erste Zeilen des Datensatzes:")
    print(df.head(10))
except FileNotFoundError:
    print(f"Fehler: Die Datei '{file_path}' wurde nicht gefunden. Bitte überprüfen Sie den Pfad und passen diesen ggf. an.")
    exit()

# Funktion zur Textbereinigung
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))  # Sonderzeichen und Zahlen entfernen
    text = text.lower()  # In Kleinbuchstaben umwandeln
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])  # Stoppwörter entfernen
    return text

# Lemmatisierung vorbereiten
lemmatizer = WordNetLemmatizer()

# Textbereinigung und Tokenisierung anwenden mit Fortschrittsbalken
tqdm.pandas(desc="Bereinige Texte")
df['cleaned_complaints'] = df['Customer Complaint'].progress_apply(clean_text)
df['tokens'] = df['cleaned_complaints'].progress_apply(
    lambda x: [lemmatizer.lemmatize(word) for word in word_tokenize(x, language='english')]
)

# Anzahl der Tokens berechnen
df['num_tokens'] = df['tokens'].apply(len)

# Ergebnisse der Bereinigung anzeigen
print("Bereinigte und tokenisierte Daten:")
print(df[['Customer Complaint', 'cleaned_complaints', 'tokens', 'num_tokens']].head())

# Funktion zum optionalen Exportieren der bereinigten Daten
def export_cleaned_data(dataframe, output_path="bereinigte_daten.csv"):
    user_choice = input("Möchten Sie die bereinigten Daten exportieren? (ja/nein): ").strip().lower()
    if user_choice == 'ja':
        cleaned_df = dataframe[['Customer Complaint', 'cleaned_complaints', 'tokens', 'num_tokens']]
        cleaned_df.to_csv(output_path, index=False)
        print(f"Bereinigte Daten wurden nach '{output_path}' exportiert.")
    else:
        print("Export wurde übersprungen.")

# Option für den Export der bereinigten Daten
export_cleaned_data(df)

# Durchschnittliche Anzahl von Tokens anzeigen
average_tokens = df['num_tokens'].mean()
print(f"Durchschnittliche Anzahl von Tokens pro Beschwerde: {average_tokens:.2f}")

# TF-IDF-Vektorisierung
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Begrenzung auf die 1000 häufigsten Wörter
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_complaints'])
print(f"TF-IDF-Matrix: {tfidf_matrix.shape}")

# Themenmodellierung mit LDA
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(tfidf_matrix)

# Schlüsselwörter für jedes Thema anzeigen
for index, topic in enumerate(lda.components_):
    print(f"Thema {index}:")
    print([tfidf_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])

# Clustering mit K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(tfidf_matrix)
df['cluster'] = kmeans.labels_

# Cluster-Verteilung anzeigen
print("Cluster-Verteilung:")
print(df['cluster'].value_counts())

# Visualisierung der Wordcloud
def generate_wordcloud(text):
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=200,
        random_state=42
    ).generate(text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Wordcloud der häufigsten Wörter in Comcast Beschwerden", fontsize=16, fontweight='bold')
    plt.show()

# Wordcloud generieren
all_text = ' '.join(df['cleaned_complaints'])
generate_wordcloud(all_text)

# Cluster-Verteilung visualisieren
sns.countplot(x='cluster', data=df)
plt.title('Verteilung der Beschwerden pro Cluster')
plt.xlabel('Cluster')
plt.ylabel('Anzahl der Beschwerden')
plt.show()