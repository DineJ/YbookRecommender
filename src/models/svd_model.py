from surprise import SVD
from surprise import accuracy
from models.base_model import BaseRecommenderModel
import logging

logger = logging.getLogger(__name__)

class SVDRecommenderModel(BaseRecommenderModel):
    """
    Recommender system based on the Singular Value Decomposition (SVD) algorithm.
    """

    def __init__(self):
        BaseRecommenderModel.__init__(self, name="SVD")
        self.model = SVD()


    def train(self, trainset):
        """
        Train the SVD model on the training set.
        """

        logger.info(f"[{self.name}] Entraînement du modèle en cours...")
        self.model.fit(trainset)
        logger.info(f"[{self.name}] Entraînement terminé.")


    def predict(self, testset):
        """
        Predict ratings for the given test set.
        """

        logger.info(f"[{self.name}] Prédictions en cours...")
        predictions = self.model.test(testset)
        logger.info(f"[{self.name}] Prédictions terminées.")
        return predictions


    def evaluate(self, predictions):
        """
        Evaluate model performance using RMSE.
        """

        logger.info(f"[{self.name}] Évaluation en cours...")
        rmse = accuracy.rmse(predictions, verbose=False)
        logger.info(f"[{self.name}] RMSE : {rmse:.4f}")
        return rmse
