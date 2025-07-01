import logging

logger = logging.getLogger(__name__)

class BaseRecommenderModel:
    """
    Abstract base class for recommender models.

    Each specific model should override the following methods:
        - train(trainset): Train the model on a training set
        - predict(testset): Generate predictions on a test set
        - evaluate(predictions): Evaluate the quality of predictions
    """

    def __init__(self, name="BaseModel"):
        """
        Initialize the model with a given name.

        Args:
            name (str): Name of the model (used for logging)
        """

        self.model = None
        self.name = name
        logger.info(f"[{self.name}] Modèle initialisé.")

    def train(self, trainset):
        """
        Train the model on the provided training set.

        Args:
            trainset: The dataset used for training

        Raises:
            NotImplementedError: Must be overridden in child class
        """

        logger.warning(f"[{self.name}] La méthode train() doit être redéfinie.")
        raise NotImplementedError("La méthode train() doit être redéfinie dans la classe enfant.")


    def predict(self, testset):
        """
        Predict results using the trained model on the test set.

        Args:
            testset: The dataset used for testing/prediction

        Raises:
            NotImplementedError: Must be overridden in child class
        """

        logger.warning(f"[{self.name}] La méthode predict() doit être redéfinie.")
        raise NotImplementedError("La méthode predict() doit être redéfinie dans la classe enfant.")


    def evaluate(self, predictions):
        """
        Evaluate the predictions made by the model.

        Args:
            predictions: The predictions to evaluate

        Raises:
            NotImplementedError: Must be overridden in child class
        """

        logger.warning(f"[{self.name}] La méthode evaluate() doit être redéfinie.")
        raise NotImplementedError("La méthode evaluate() doit être redéfinie dans la classe enfant.")
