# YbookRecommender

Application de recommandation personnalisée de livres basée sur les notes des utilisateurs.  
Cette application utilise plusieurs modèles de filtrage collaboratif pour prédire les livres qu'un utilisateur pourrait aimer.

---

## Structure du projet

- `src/` : Contient le code source de l'application  
  - `app.py` : Point d'entrée de l'application Streamlit  
  - `models/` : Code relatif à l'entraînement, la sauvegarde et le chargement des modèles  
  - `datasetcleaner/` : Classes pour nettoyer et préparer les différents jeux de données  
  - `recommender.py` : Script pour entraîner et sélectionner le meilleur modèle  

- `datas/` : Données  
  - `csv/` : Contient les fichiers CSV bruts (ex : books.csv, ratings.csv, tags.csv, etc.)  
  - `pickle/` : Stocke les modèles entraînés (ex : best_model_SVD.pkl)  

---

## Prérequis

- Python 3.10  
- Environnement virtuel recommandé  

---

## Installation

1. Cloner le dépôt :
    ```bash
    git clone https://github.com/DineJ/YbookRecommender.git
    cd YbookRecommender
    ```

2. Créer et activer un environnement virtuel (optionnel mais recommandé) :
    ```bash
    python -m venv env310
    # Sous Windows
    .\env310\Scripts\activate
    # Sous Linux/MacOS
    source env310/bin/activate
    ```

3. Installer les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

---

## Utilisation

### Entraîner et sauvegarder le modèle

Lancer le script `recommender.py` qui entraîne plusieurs modèles, sélectionne le meilleur, puis le sauvegarde dans `datas/pickle/`.

```bash
python src/recommender.py
