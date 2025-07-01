# Librairies

from datasetcleaner.datasetcleanergeneric import DatasetCleanerGeneric
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)



class DatasetBooksCleaner(DatasetCleanerGeneric):
    """
    Cleans book-related datasets using domain-specific rules

    Inherits from DatasetCleanerGeneric and adds targeted cleaning methods
    """

    # Cleaning books
    def clean(self):
        """
        Cleans the books DataFrame by removing unnecessary columns and filtering invalid data.

        - Drops unused columns such as ISBN and image URLs
        - Filters books with only positive identifiers and valid ISBN13 prefixes
        - Resets index after filtering

        Returns:
            DatasetBooksCleaner: The instance with the cleaned DataFrame.
        """

        try:
            # Count lines before filtering
            before = len(self.df)

            # Delete useless column
            self.df = self.df.drop(['isbn', 'small_image_url', 'work_ratings_count'], axis=1, errors='ignore')

            # Values over 0 for book_id, best_book_id, work_id and books_count
            self.df = self.df[(self.df['book_id'] > 0) & (self.df['best_book_id'] > 0) & (self.df['work_id'] > 0) & (self.df['books_count'] > 0)]

            # Values start with 978 or 979 and contains 13 numbers
            self.df = self.df[(self.df['isbn13'] // 10**10).isin([978, 979])]

            # Reset index
            self.df = self.df.reset_index(drop=True)

            # Display size of dataframe before and after filtering
            self.dataframe_size(before,1, name=self.name)

        except Exception as e:
            logger.exception("Erreur dans clean_books")
        return self


    def data_wishlist_rating(self, to_read_df):
        """
        Prepares datasets for plotting wishlist-related bar charts.

        This method generates two data dictionaries:
        1. Top 10 most wished-for books (based on `to_read_df`)
        2. Top 10 highest-rated books (with at least 100 ratings)

        Args:
            to_read_df (pd.DataFrame): A DataFrame containing book IDs from users' to-read lists.

        Returns:
            list of dict: Each dict contains data and plot configuration (e.g., x/y columns, titles)
            ready to be used for bar chart generation.
        """

        # Get most wished book
        wishlist_counts = self.merge_count(to_read_df, 'book_id', 'left', 'title', 'wishlist_counts', 10, ['book_id', 'title'])

        # Filter books with atleast 100 rates
        filtered_df = self.df[self.df['ratings_count'] >= 100]
        most_rated = self.sort('ratings_count', 10, filtered_df)

        # Best books out of 10
        top_rated = self.sort('average_rating', 10, most_rated)

        return (
            self.create_dic(top_rated, 'average_rating', 'title', 'Top 10 des livres les mieux notés', 'Note moyenne', '', 'Greens'),
            self.create_dic(wishlist_counts, 'wishlist_counts', 'title', 'Top 10 des livres les plus souhaités', "Nombre d'ajouts", '', 'Reds')
        )


    def data_average_rank_by_reader(self, ratings_df, books_df):
        """
        Compute data for plots:
        - Average rating given by the top 10 users (by number of ratings)
        - Average deviation from book average rating by these top users
        
        Args:
            ratings_df (pd.DataFrame): DataFrame with at least ['user_id', 'book_id', 'rating']
            books_df (pd.DataFrame): DataFrame with at least ['book_id', 'title']
        
        Returns:
            tuple: (mean_ratings_df, mean_diff_per_user_df)
                mean_ratings_df: avg rating per top user (user_id, rating)
                mean_diff_per_user_df: avg diff between user rating and book avg rating (user_id, rating_diff)
        """
        # Top 10 users by number of ratings
        top_users = ratings_df.groupby('user_id').size().sort_values(ascending=False).head(10).index

        # Ratings from top users merged with book titles (optional)
        top_users_ratings = ratings_df[ratings_df['user_id'].isin(top_users)]
        top_users_ratings = top_users_ratings.merge(books_df[['book_id', 'title']], on='book_id', how='left')

        # Average rating given by each top user
        mean_ratings = top_users_ratings.groupby('user_id')['rating'].mean().reset_index()

        # Average rating per book
        average_book_ratings = ratings_df.groupby('book_id')['rating'].mean().reset_index(name='average_rating')

        # Merge average book rating with all ratings
        ratings_with_avg = ratings_df.merge(average_book_ratings, on='book_id')

        # Compute difference between user rating and average book rating
        ratings_with_avg['rating_diff'] = ratings_with_avg['rating'] - ratings_with_avg['average_rating']

        # Filter for top users
        top_users_diff = ratings_with_avg[ratings_with_avg['user_id'].isin(top_users)]

        # Average rating difference per top user
        mean_diff_per_user = top_users_diff.groupby('user_id')['rating_diff'].mean().reset_index()

        return (
            self.create_dic(mean_ratings, 'user_id', 'rating', 'Top 10 lecteurs les plus actifs - note moyenne', 'Note moyenne', 'ID utilisateur', 'Blues'),
            self.create_dic(mean_diff_per_user, 'user_id', 'rating_diff', 'Top 10 lecteurs les plus actifs - écart moyen à la note globale', 'Écart moyen', 'ID utilisateur', 'Oranges')
        )


    def data_heatmap(self):
        # Heatmap showing the correlations between key book features: ratings count, average rating, and review count
        # Select relevant columns for correlation analysis
        cols = ['ratings_count', 'average_rating', 'work_text_reviews_count']
        books_clean_graph = self.df[cols]

        # Compute the correlation matrix for the selected features
        return books_clean_graph.corr()