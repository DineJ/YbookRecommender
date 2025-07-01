import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy


from datasetcleaner.datasetbookscleaner import DatasetBooksCleaner
from datasetcleaner.datasettagscleaner import DatasetTagsCleaner
from datasetcleaner.datasetbookstagscleaner import DatasetBooksTagsCleaner
from datasetcleaner.datasetratingscleaner import DatasetRatingsCleaner
from datasetcleaner.datasettoreadcleaner import DatasetToReadCleaner

class Graphic:

    # Constructor
    def __init__(self):
        """
        Initialize the Graphic object.
        """

        # Initialize DataCleaner instances for each dataset
        self.books = DatasetBooksCleaner("books")
        self.tags = DatasetTagsCleaner("tags")
        self.book_tags = DatasetBooksTagsCleaner("book_tags")
        self.ratings = DatasetRatingsCleaner("ratings")
        self.to_read = DatasetToReadCleaner("to_read")
        
        # Clean each DataCleaner instances
        self.books.clean()
        self.tags.clean()
        self.book_tags.clean(self.tags.df, self.books.df)
        self.ratings.clean(self.books.df)
        self.to_read.clean(self.books.df)

        # Create default size for plot
        self.width = 24
        self.height = 8


    # Display graph
    def barplot_wishlist(self):
        """
        Display barplots for user wishlist statistics.

        This method generates one or multiple side-by-side barplots using
        data returned by the `data_wishlist_rating` method from the cleaned books dataset.
        It visualizes relationships such as book popularity or rating in users' to-read lists.
        """

        # Generate plot
        datasets = self.books.data_wishlist_rating(self.to_read.df)
        number_graph = len(datasets)

        _, axes = plt.subplots(1, number_graph, figsize=(self.width, self.height))

        # Ensure axes is iterable
        if number_graph == 1:
            axes = [axes]

        # Loop through each dataset and plot corresponding graph
        for i, params in enumerate(datasets):
            # Draw a barplot using given data and parameters
            sns.barplot(data=params['data'],x=params['x'],y=params['y'], palette=params.get('palette'),ax=axes[i])
            axes[i].tick_params(axis='x')
            axes[i].set_title(params.get('title', f"Graph {i+1}"))
            axes[i].set_xlabel(params.get('axis_name_x'))
            axes[i].set_ylabel(params.get('axis_name_y'))
                                                                                                                                                                                                                                                     
        plt.tight_layout()
        plt.show()

    def barplot_ratings(self):
        """
        Display barplots for rating statistics by top readers.

        This method generates side-by-side barplots using data returned by
        the `data_average_rank_by_reader` method. It visualizes:
        1) The average rating given by the top 10 users (by number of ratings)
        2) The average deviation from the overall book ratings by these users
        """

        # Retrieve formatted datasets
        datasets = self.books.data_average_rank_by_reader(self.ratings.df, self.books.df)
        number_graph = len(datasets)

        _, axes = plt.subplots(1, number_graph, figsize=(self.width, self.height))

        # Ensure axes is iterable
        if number_graph == 1:
            axes = [axes]

        # Loop through each dataset and render barplots
        for i, params in enumerate(datasets):
            sns.barplot(
                data=params['data'],
                x=params['x'],
                y=params['y'],
                palette=params.get('palette', 'Set2'),
                ax=axes[i]
            )

            # Configure axis labels and title
            axes[i].set_title(params.get('title', f"Graph {i+1}"))
            axes[i].set_xlabel(params.get('axis_name_x', ''))
            axes[i].set_ylabel(params.get('axis_name_y', ''))

            # Rotate x labels for clarity
            axes[i].tick_params(axis='x', rotation=0)

            # Annotate bars with values rounded to 1 decimal
            for idx, row in params['data'].iterrows():
                value = row[params['y']]
                axes[i].text(
                    x=idx,
                    y=value + 0.05,
                    s=f"{round(value, 1)}",
                    ha='center',
                    fontsize=9
                )

            # Add horizontal line at 0 for deviation plots
            if 'diff' in params['y'].lower():
                axes[i].axhline(0, color='gray', linestyle='--')

        plt.tight_layout()
        plt.show()


    def barplot_tag(self):
        """
        Display barplot for the most frequently used tags.

        This method fetches tag frequency data using the `data_most_tag` method,
        and renders a bar chart showing the top 20 most common tags applied to books.
        """

        # Get dataset formatted for plotting
        datasets = [self.tags.data_most_tag(self.book_tags.df)]
        number_graph = len(datasets)

        _, axes = plt.subplots(1, number_graph, figsize=(self.width, self.height))

        # Ensure axes is iterable
        if number_graph == 1:
            axes = [axes]

        # Plot the tag frequency barplot
        for i, params in enumerate(datasets):
            sns.barplot(data=params['data'],x=params['x'],y=params['y'], palette=params.get('palette'),ax=axes[i])
            axes[i].tick_params(axis='x')
            axes[i].set_title(params.get('title'))
            axes[i].set_xlabel(params.get('axis_name_x'))
            axes[i].set_ylabel(params.get('axis_name_y'))

        plt.tight_layout()
        plt.show()

    def heatmap(self):
    
        corr = self.books.data_heatmap()

        # Create a mask to hide the upper triangle of the symmetric correlation matrix for better readability
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Plot the heatmap with annotations and apply the mask to show only the lower triangle
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", mask=mask)

        plt.title('Heatmap liant la note moyenne, le nombre de vote et le nombre de revue')

        # Rotate y-axis labels to horizontal and align text to the right for clarity
        plt.yticks(rotation=0, ha='right')

        plt.show()
        
    def streamlit(self, pipeline):
        """
        Lance l'application Streamlit.
        
        Args:
            pipeline: instance de RecommenderPipeline avec la méthode recommend_for_user.
        """

        st.set_page_config(page_title="Recommandation de livres", layout="wide")
        st.title("Recommandations personnalisées")

        # Entrée utilisateur
        user_ids = sorted(self.ratings.df['user_id'].unique())
        user_id = st.selectbox("Choisissez un utilisateur", user_ids)

        n_recos = st.slider("Nombre de recommandations", min_value=5, max_value=20, value=10)

        if st.button("Afficher recommandations"):
            with st.spinner("Calcul des recommandations..."):
                recommandations = pipeline.recommend_for_user(user_id=user_id, n=n_recos)

                if recommandations.empty:
                    st.warning("Aucune recommandation trouvée pour cet utilisateur.")
                else:
                    st.subheader(f"Top {n_recos} recommandations pour l'utilisateur {user_id}")
                    st.dataframe(recommandations[['title', 'predicted_rating', 'authors']])

                    recommandations_sorted = recommandations.sort_values('predicted_rating', ascending=True)

                    fig = px.bar(
                        recommandations_sorted,
                        x='predicted_rating',
                        y='title',
                        orientation='h',
                        hover_data=['authors'],
                        text='predicted_rating',
                        title=f'Recommandations pour l’utilisateur {user_id}',
                        labels={'predicted_rating': 'Note prédite', 'title': 'Titre du livre'}
                    )

                    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                    fig.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        xaxis=dict(range=[
                            recommandations['predicted_rating'].min() - 0.5,
                            recommandations['predicted_rating'].max() + 0.5
                        ])
                    )
                    st.plotly_chart(fig, use_container_width=True)

