# Librairies

from datasetcleaner.datasetcleanergeneric import DatasetCleanerGeneric
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class DatasetRatingsCleaner(DatasetCleanerGeneric):
    """
    Cleans ratings-related datasets using domain-specific rules

    Inherits from DatasetCleanerGeneric and adds targeted cleaning methods
    """

    # Cleaning ratings
    def clean(self, books_clean):
        """
        Cleans the ratings DataFrame by validating book and user IDs and rating values.

        - Removes ratings with `book_id` not found in `books_clean`
        - Keeps only ratings with values between 1 and 5
        - Filters out rows where `book_id` or `user_id` is not positive
        - Resets the index after filtering

        Args:
            books_clean (pd.DataFrame): Cleaned books DataFrame to validate `book_id` references

        Returns:
            DatasetRatingsCleaner: The instance with the cleaned DataFrame stored in `self.df`
        """

        try:
            # Count lines before filtering
            before = len(self.df)

            # Delete books_id without match
            self.drop_invalid_ids('book_id',books_clean['book_id'])
            logger.info("Valeurs sans correspondance dans books_clean pour ratings :")
            logger.debug(f"{self.df['book_id'].unique()}")

            # Values are between 1 and 5 for rating
            self.df = self.df[(self.df['rating'] > 0) & (self.df['rating'] < 6)]

            # Values over 0 for book_id and user_id
            self.df = self.filter_positive(['book_id', 'user_id'])

            # Reset index
            self.df = self.df.reset_index(drop=True)

            message = f"Lignes supprimées dans le Dataframe {self.name}"
            # Display size of dataframe before and after filtering
            self.dataframe_size(before,1, name=self.name)

        except Exception as e:
            logger.exception("Erreur dans ratings_clean")
        return self