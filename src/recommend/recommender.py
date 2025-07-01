import logging
import os
import pandas as pd
import pickle
from surprise import SVD, KNNBasic, accuracy, NMF, BaselineOnly, Dataset, Reader
from surprise.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class RecommenderPipeline:
    def __init__(self, ratings_df, books_df, model_path="..\\datas\\pickle\\best_model_SVD.pkl"):
        self.ratings_df = ratings_df
        self.books_df = books_df
        self.model_path = model_path
        self.best_model = None
        self.best_model_name = ""
        self.best_rmse = float('inf')

        # Assure-toi que le dossier "../datas/pickle" existe
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # Charger ou entraîner le modèle
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.best_model = self.model  # 👈 ajoute cette ligne
            print(f"Modèle chargé depuis {self.model_path}")

        else:
            print("Entraînement du modèle SVD...")
            self.model = self.train_model()
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Modèle sauvegardé dans {self.model_path}")

    def train_model(self):
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

        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.ratings_df[['user_id', 'book_id', 'rating']], reader)
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

        for name, model in models.items():
            logger.info(f"Entraînement du modèle : {name}")
            model.fit(trainset)
            predictions = model.test(testset)
            rmse = accuracy.rmse(predictions, verbose=False)
            mae = accuracy.mae(predictions, verbose=False)
            results[name] = {"rmse": rmse, "mae": mae}
            logger.info(f"{name} - RMSE : {rmse:.4f} | MAE : {mae:.4f}")

            if rmse < self.best_rmse:
                self.best_rmse = rmse
                self.best_model = model
                self.best_model_name = name

        logger.info(f"Meilleur modèle : {self.best_model_name} avec une RMSE de : {self.best_rmse:.4f}")
        return self.best_model

    def recommend_for_user(self, user_id: int, n: int = 10) -> pd.DataFrame:
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
