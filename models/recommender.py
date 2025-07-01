import logging
import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic, NMF, accuracy
from surprise.model_selection import train_test_split
import pickle
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RecommenderPipeline:
    def __init__(self, ratings_df: pd.DataFrame, books_df: pd.DataFrame):
        """
        Initialise le pipeline de recommandation avec les dataframes de notes et de livres.

        Args:
            ratings_df (pd.DataFrame): DataFrame contenant les colonnes ['user_id', 'book_id', 'rating']
            books_df (pd.DataFrame): DataFrame avec les métadonnées des livres, doit contenir 'book_id'
        """

        self.ratings_df = ratings_df
        self.books_df = books_df
        self.reader = Reader(rating_scale=(1, 5))
        self.data = Dataset.load_from_df(self.ratings_df[['user_id', 'book_id', 'rating']], self.reader)
        self.trainset, self.testset = train_test_split(self.data, test_size=0.2, random_state=42)
        self.best_model = None
        self.best_model_name = ""
        self.best_rmse = float('inf')


    def train_and_evaluate(self):
        """
        Entraîne plusieurs modèles et les évalue sur le jeu de test.
        Stocke le meilleur modèle basé sur la RMSE.
        """

        models = {
            "SVD": SVD(),
            "KNNBasic": KNNBasic(sim_options={'name': 'cosine', 'user_based': False}),
            "NMF": NMF()
        }

        results = {}

        for name, model in models.items():
            logger.info(f"Entraînement du modèle : {name}")
            model.fit(self.trainset)
            predictions = model.test(self.testset)
            rmse = accuracy.rmse(predictions, verbose=False)
            mae = accuracy.mae(predictions, verbose=False)
            results[name] = {"rmse": rmse, "mae": mae}
            logger.info(f"{name} - RMSE : {rmse:.4f} | MAE : {mae:.4f}")

            if rmse < self.best_rmse:
                self.best_rmse = rmse
                self.best_model = model
                self.best_model_name = name

        logger.info(f"Meilleur modèle : {self.best_model_name} avec une RMSE de : {self.best_rmse:.4f}")


    def save_best_model(self, filepath='models/'):
        """
        Sauvegarde le meilleur modèle au format pickle dans le chemin donné.
        """

        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filename = os.path.join(filepath, f"best_model_{self.best_model_name}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(self.best_model, f)
        logger.info(f"Meilleur modèle sauvegardé dans : {filename}")


    def recommend_for_user(self, user_id, n=10):
        """
        Génère les N meilleures recommandations de livres pour un utilisateur donné en utilisant le meilleur modèle entraîné.

        Args:
            user_id (int): L'identifiant de l'utilisateur pour lequel générer les recommandations.
            n (int): Nombre de recommandations à générer.

        Returns:
            pd.DataFrame: DataFrame contenant les livres recommandés avec leurs notes prédites.
        """

        all_books = self.books_df['book_id'].unique()
        read_books = self.ratings_df[self.ratings_df['user_id'] == user_id]['book_id'].unique()
        unseen_books = [book for book in all_books if book not in read_books]

        predictions = [self.best_model.predict(user_id, book_id) for book_id in unseen_books]
        top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]

        recommended_books = pd.DataFrame({
            'book_id': [pred.iid for pred in top_predictions],
            'predicted_rating': [pred.est for pred in top_predictions]
        })

        return recommended_books.merge(self.books_df, on='book_id')
