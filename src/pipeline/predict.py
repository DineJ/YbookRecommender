import pandas as pd
from models.recommender import RecommenderSystem
from src.utils.recommendation_utils import get_top_n_recommendations
from src.config import RATINGS_PATH, BOOKS_PATH, BEST_MODEL_PATH

ratings = pd.read_csv(RATINGS_PATH)
books = pd.read_csv(BOOKS_PATH)

recommender = RecommenderSystem(None, "LoadedModel")
recommender.load(BEST_MODEL_PATH)

user_id = 42
recommendations = get_top_n_recommendations(user_id, recommender.model, ratings, books)

print(f"\nRecommandations pour l'utilisateur {user_id} :")
print(recommendations[['title', 'predicted_rating']])

