# NVIDIA Sentiment Analysis Project

Ce projet analyse les sentiments financiers concernant NVIDIA en utilisant FinBERT, un modèle NLP spécialisé dans le domaine financier.

## Objectif

Évaluer les sentiments (positif, neutre, négatif) des actualités et publications financières liées à NVIDIA et observer la corrélation avec l'évolution du cours de l'action.

## Démo en ligne

Accédez à la version déployée du dashboard ici :
🔗 [nvidia-sentiment-analysiss.streamlit.app](https://nvidia-sentiment-analysiss.streamlit.app/)

## Structure du projet

```
NVIDIA/
├── app.py                 # Application Streamlit principale
├── data_collector.py      # Collecte des données d'actualités
├── sentiment_analyzer.py  # Analyse de sentiment avec FinBERT
├── stock_data.py         # Récupération des données boursières
├── visualizer.py         # Création des graphiques et visualisations
├── utils.py              # Fonctions utilitaires
├── requirements.txt      # Dépendances Python
├── config.py            # Configuration du projet
├── data/                # Dossier pour stocker les données
│   ├── raw/            # Données brutes
│   └── processed/      # Données traitées
└── README.md           # Ce fichier
```

## Installation

1. Créer un environnement virtuel :
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Lancer l'application :
```bash
streamlit run app.py
```

## Fonctionnalités

- **Collecte automatique de données** : Récupération d'articles de presse et tweets sur NVIDIA
- **Analyse de sentiment** : Utilisation du modèle FinBERT pour analyser les sentiments
- **Visualisations interactives** : Graphiques de sentiment vs cours boursier
- **Dashboard Streamlit** : Interface utilisateur intuitive
- **Corrélation sentiment-prix** : Analyse statistique des relations

## Technologies utilisées

- Python, Pandas, NumPy
- Transformers (FinBERT)
- Streamlit
- Matplotlib, Seaborn, Plotly
- yfinance pour les données boursières
- BeautifulSoup pour le web scraping

