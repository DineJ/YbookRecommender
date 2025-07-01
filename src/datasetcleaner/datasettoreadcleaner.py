# Librairies

from datasetcleaner.datasetcleanergeneric import DatasetCleanerGeneric
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class DatasetToReadCleaner(DatasetCleanerGeneric):
    """
    Cleans to_read-related datasets using domain-specific rules

    Inherits from DatasetCleanerGeneric and adds targeted cleaning methods
    """

    # Cleaning to_read
    def clean(self, books_clean):
        """
        Cleans the to-read DataFrame using the provided cleaned books dataset.

        - Removes entries with unmatched `book_id` values not found in `books_clean`
        - Filters out rows with non-positive `book_id` or `user_id`
        - Resets the index after filtering

        Args:
            books_clean (pd.DataFrame): Cleaned books DataFrame used to validate book_id entries

        Returns:
            DatasetToReadCleaner: The instance with the cleaned DataFrame stored in `self.df`
        """

        try:
            # Count lines before filtering
            before = len(self.df)

            # Delete books_id without match
            self.drop_invalid_ids('book_id', books_clean['book_id'])
            logger.info("Valeurs sans correspondance dans books_clean pour to_read :")
            logger.debug(f"{self.df['book_id'].unique()}")

            # Values over 0 for book_id and user_id
            self.df = self.df[(self.df['book_id'] > 0) & (self.df['user_id'] > 0)]

            # Reset index
            self.df = self.df.reset_index(drop=True)

            # Display size of dataframe before and after filtering
            self.dataframe_size(before,1, name=self.name)

        except Exception as e:
            logger.exception("Erreur to_read_clean")
        return self
