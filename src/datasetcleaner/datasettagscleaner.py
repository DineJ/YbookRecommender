# Librairies

from datasetcleaner.datasetcleanergeneric import DatasetCleanerGeneric
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class DatasetTagsCleaner(DatasetCleanerGeneric):
    """
    Cleans tags-related datasets using domain-specific rules

    Inherits from DatasetCleanerGeneric and adds targeted cleaning methods
    """

    # Cleaning tags
    def clean(self):
        """
        Cleans the tags DataFrame by filtering and formatting the data.

        - Filters out rows with non-positive `tag_id`
        - Normalizes `tag_name` by removing extra dashes and whitespace
        - Resets the index after cleaning

        Returns:
            DatasetTagsCleaner: The instance with the cleaned DataFrame stored in `self.df`
        """

        try:
            # Count lines before filtering
            before = len(self.df)

            # Values are over 0 for tag_id
            self.df = self.df[self.df['tag_id'] > 0]

            # Replace every "-" by " "
            self.df['tag_name'] = self.df['tag_name'].str.replace(r'^-+|-+$|-+', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()

            # Reset index
            self.df = self.df.reset_index(drop=True)

            # Display size of dataframe before and after filtering
            self.dataframe_size(before,1, name=self.name)

        except Exception as e:
            logger.exception("Erreur dans clean_book_tag")
        return self


    def data_most_tag(self, books_tags_df):
        # Barplot displaying the top 20 most frequent tags in the book catalog
        # Count the occurrences of each tag (assuming 'book_tags_clean' links books and tags)

        tag_counts = books_tags_df.groupby('tag_id').size().reset_index(name='count')

        # Merge with tag names
        tag_counts = tag_counts.merge(self.df[['tag_id', 'tag_name']], on='tag_id', how='left')

        # Sort and keep the top 20 most frequent tags
        # top_tags = tag_counts.sort_values(by='count', ascending=False).head(20)
        top_tags = self.sort('count', 20, tag_counts)
        return self.create_dic(top_tags, 'tag_name', 'tag_id',  'Les 20 tags les plus courants', 'Tags', 'Nombre apparition', 'viridis')