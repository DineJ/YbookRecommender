# Librairies

from datasetcleaner.datasetcleanergeneric import DatasetCleanerGeneric
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class DatasetBooksTagsCleaner(DatasetCleanerGeneric):
    """
    Cleans book_tags-related datasets using domain-specific rules

    Inherits from DatasetCleanerGeneric and adds targeted cleaning methods
    """

    # Cleaning book_tag
    def clean(self,tags_clean,books_clean):
        """
        Cleans the book-tags mapping DataFrame using reference data.

        - Renames columns for standardization (e.g. `goodreads_book_id` to `book_id`)
        - Removes rows with `tag_id` or `book_id` not found in `tags_clean` or `books_clean`
        - Filters out entries where `tag_id`, `count`, or `book_id` are non-positive
        - Resets the index after filtering

        Args:
            tags_clean (pd.DataFrame): Cleaned tags DataFrame for validating `tag_id`
            books_clean (pd.DataFrame): Cleaned books DataFrame for validating `book_id`

        Returns:
            DatasetBooksTagsCleaner: The instance with the cleaned DataFrame in `self.df`
        """

        try:
            # Count lines before filtering
            before = len(self.df)

            # Standardizing column name
            rename_map = {"goodreads_book_id": "book_id"}
            self.df.rename(columns=rename_map, inplace=True)

            # Delete tag_id without match
            self.drop_invalid_ids('tag_id',tags_clean['tag_id'])
            logger.info("Valeurs sans correspondance dans tags_clean :")
            logger.debug(f"{self.df['tag_id'].unique()}")

            # Delete book_id without match
            self.drop_invalid_ids('book_id',books_clean['book_id'])
            logger.info("Valeurs sans correspondance dans books_clean :")
            logger.debug(f"{self.df['book_id'].unique()}")

            # Values are over 0 for tag_id, count and goodreads_book_id
            self.df = self.df[(self.df['tag_id'] > 0) & (self.df['count'] > 0) & (self.df['book_id'] > 0)]

            # Reset index
            self.df = self.df.reset_index(drop=True)

            # Display size of dataframe before and after filtering
            self.dataframe_size(before,1, name=self.name)

        except Exception as e:
            logger.exception("Erreur dans book_tags")
        return self