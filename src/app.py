from graph.graphic import Graphic
from recommend.recommender import RecommenderPipeline

if __name__ == "__main__":
    # Initialiser Graphic => cela nettoie les données
    graphic = Graphic()

    # Créer le pipeline de recommandation
    pipeline = RecommenderPipeline(
        ratings_df=graphic.ratings.df,
        books_df=graphic.books.df
    )
    # Lancer l’interface
    graphic.streamlit(pipeline)

