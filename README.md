# Comcast Complaint Analysis Project

## Beschreibung

Dieses Projekt analysiert Beschwerden von Comcast-Kunden basierend auf einem bereitgestellten Datensatz. Die Hauptziele sind:
1. **Textbereinigung**: Entfernen von Sonderzeichen, Stoppwörtern und Zahlen sowie Tokenisierung und Lemmatisierung.
2. **Themenmodellierung**: Identifikation von Schlüsselthemen in den Beschwerden mit LDA.
3. **Clustering**: Gruppierung ähnlicher Beschwerden mit K-Means.
4. **Visualisierungen**: Darstellung der Ergebnisse mittels Wordcloud und Cluster-Visualisierung.

---

## Voraussetzungen

Um dieses Projekt auszuführen, benötigen Sie folgende Software:
- **Python**: Version 3.10 oder neuer.
- **Virtuelle Umgebung (empfohlen)**: Verwendung von `venv` oder einer ähnlichen Umgebung.

Die notwendigen Python-Bibliotheken sind in der Datei `requirements.txt` aufgelistet.

---

## Installation

1. Klonen Sie das Repository:
   ```bash
   git clone https://github.com/lv082599/ComcastComplaintAnalysis.git
   cd ComcastComplaintAnalysis
