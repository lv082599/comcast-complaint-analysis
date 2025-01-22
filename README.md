# Comcast Complaint Analysis Project

## Beschreibung

Dieses Projekt analysiert Beschwerden von Comcast-Kunden basierend auf einem bereitgestellten Datensatz. Die Hauptziele sind:
1. **Textbereinigung**: Entfernen von Sonderzeichen, Stoppwörtern und Zahlen sowie Tokenisierung und Lemmatisierung.
2. **Themenmodellierung**: Identifikation von Schlüsselthemen in den Beschwerden mit LDA und NMF.
3. **Clustering**: Gruppierung ähnlicher Beschwerden mit K-Means und Visualisierung mittels PCA.
4. **Visualisierungen**: Darstellung der Ergebnisse mittels Wordcloud und Cluster-Visualisierung.

---

## Voraussetzungen

Um dieses Projekt auszuführen, wird folgende Python-Version empfohlen:
- **Python**: Version 3.13.3
- **Virtuelle Umgebung (empfohlen)**: Verwendung von `venv`.

Die notwendigen Python-Bibliotheken sind in der Datei `requirements.txt` aufgelistet.

---

## Installation

### Repository klonen
```bash
git clone https://github.com/lv082599/comcast-complaint-analysis.git
cd comcast-complaint-analysis
```

### Virtuelle Umgebung erstellen und aktivieren
```bash
python3 -m venv venv
source venv/bin/activate  # (Mac)
venv\Scripts\activate     # (Windows CMD)
```

### Abhängigkeiten installieren
```bash
pip3 install -r requirements.txt
```

---

## Ausführung

### Skript ausführen
```bash
python3 main.py
```

### Ergebnisse
1. **Bereinigungsergebnisse**: Die bereinigten Texte werden angezeigt und optional als CSV exportiert.
2. **Themenmodellierung**: Ergebnisse von LDA und NMF werden in der Konsole ausgegeben und visualisiert.
3. **Cluster-Analyse**: Cluster-Verteilung wird mit Balkendiagrammen und PCA-Visualisierungen dargestellt.
4. **Wordcloud**: Häufige Wörter werden in einer Wordcloud dargestellt.

---

## Projektstruktur
- `main.py`: ist das Hauptskript zur Durchführung der Analyse.
- `requirements.txt`: Liste der benötigten Bibliotheken.
- `Comcast.csv`: Beispiel-Datensatz mit Beschwerden.

---

## Features
- **Textbereinigung**: Entfernt unnötige Informationen und fokussiert sich auf die Kerninhalte der Beschwerden.
- **Vergleichende Themenmodellierung**: Anwendung von LDA und NMF zur Analyse der Hauptthemen.
- **Cluster-Analyse**: Gruppierung von Beschwerden für eine effizientere Bearbeitung.
- **Visualisierungen**: Einfache Interpretation der Ergebnisse durch Diagramme und Wordclouds.

---

## Beispiele

### Beispiel-Themenmodellierung
- **Thema 0 (LDA)**: ["service", "internet", "speed", "problem", "slow"]
- **Thema 1 (NMF)**: ["bill", "charge", "payment", "overcharged", "refund"]

### Cluster-Verteilung
- Cluster 0: 40% der Beschwerden
- Cluster 1: 30% der Beschwerden

---

## Kontakt

Für Fragen oder Anmerkungen wenden Sie sich bitte an:
- **Name**: Lukas Vitzthum
- **GitHub**: [lv082599](https://github.com/lv082599)
- **E-Mail**: lukas.vitzthum@example.com
