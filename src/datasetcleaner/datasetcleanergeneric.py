# Librairies

import pandas as pd
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class DatasetCleanerGeneric:

    MSG_MISSING_VALUES = 1
    MSG_INVALID_VALUES = 2
    MSG_NO_MATCHING_IDS = 3
    MSG_DUPLICATES_VALUES = 4

    """
    Generic class for cleaning CSV files containing tabular data

    This class loads a CSV file from a given path and applies basic cleaning steps
    such as removing duplicates, handling missing values, and normalizing column names

    Attributes:
        name (str): Name of the file without extension
        path (str): Directory path where the CSV file is located
        df (pd.DataFrame): Loaded data as a pandas DataFrame
    """

    # Constructor
    def __init__(self, name, path='../datas/csv/'):
        """
        Initializes the DatasetCleanerGeneric class by loading the CSV file
        and performing initial data cleaning steps

        Args:
            name (str): Filename without the `.csv` extension
            path (str): Path to the folder containing the file (default is '../datas/csv/')

        Raises:
            FileNotFoundError: If the CSV file does not exist
            pd.errors.ParserError: If the CSV file is corrupted or improperly formatted
            Exception: For any other unexpected errors
        """

        self.name = name
        try:
            filepath = os.path.join(path, f"{self.name}.csv")
            self.df = pd.read_csv(filepath)

        except FileNotFoundError:
            logger.error(f"Erreur : fichier {self.name} non trouvé.")
            self.df = pd.DataFrame()
            return

        except pd.errors.ParserError:
            logger.error(f"Erreur : Le fichier {self.name}.csv est corrompu ou mal formaté.")
            self.df = pd.DataFrame()
            return

        except Exception as e:
            logger.exception("Erreur inattendue lors du chargement du fichier")
            self.df = pd.DataFrame()
            return

        try:
            logger.debug(f"[INIT] Colonnes chargées dans {self.name} : {self.df.columns.tolist()}")
            self.data_type()
            self.data_duplicate()
            self.data_spaces()
            self.data_nan()
            self.is_empty()

        except Exception as e:
            logger.exception("Erreur lors du nettoyage du dataset")


    # Display data's type
    def data_type(self):
        """
        Identifies columns containing mixed data types

        Iterates through all columns of the DataFrame and logs any
        columns that contain more than one unique data type

        Returns:
            None
        """

        try:
            logger.debug(f"[INIT.data_type] Colonnes chargées dans {self.name} : {self.df.columns.tolist()}")
            colonne_errone = []
            logger.info(f"Affichage des colonnes de {self.name}.csv ayant plusieurs types de données")

            # Loop through each column and check for multiple data types
            for colonne in self.df.columns:
                types_uniques = self.df[colonne].map(type).value_counts()
                if len(types_uniques) > 1:
                    colonne_errone.append(colonne)
                    logger.warning(f"Colonne '{colonne}' contient plusieurs types \n")
                    for t, count in types_uniques.items():
                        logger.warning(f"  - {t.__name__}: {count} lignes\n")

            if not colonne_errone:
                logger.info("Aucune colonne avec types mixtes détectée\n")

        except Exception as e:
            logger.exception("Erreur dans data_type")


    # Remove duplicates
    def data_duplicate(self):
        """
        Identifies and removes duplicated rows from the DataFrame

        Logs the number of duplicated rows found and deletes them

        Returns:
            None
        """

        try:
            logger.debug(f"[INIT.data_duplicate] Colonnes chargées dans {self.name} : {self.df.columns.tolist()}")
            logger.info("Vérification des doublons\n")
            nb_doublons = self.df.duplicated().sum()
            logger.info(f"{nb_doublons} doublons détectés dans {self.name}.csv\n")

            if nb_doublons > 0:
                # Count lines before filtering
                before = len(self.df)
                # replace original dataframe with cleaned version
                self.df = self.df.drop_duplicates()
                self.dataframe_size(before,4, name=self.name)

        except Exception as e:
            logger.exception("Erreur dans data_duplicate")


    # Delete spaces
    def data_spaces(self):
        """
        Removes leading and trailing spaces from column names and converts them to lowercase

        Returns:
            None
        """

        try:
            logger.debug(f"[INIT.data_spaces] Colonnes chargées dans {self.name} : {self.df.columns.tolist()}")
            self.df.columns = self.df.columns.str.strip().str.lower()

        except Exception as e:
            logger.exception("Erreur dans data_spaces") 


    # Remove missing values
    def data_nan(self):
        """
        Removes rows that contain at least one NaN value

        Logs the number of rows removed and resets the index afterward

        Returns:
            None
        """

        try:
            logger.debug(f"[INIT.data_nan] Colonnes chargées dans {self.name} : {self.df.columns.tolist()}")
            logger.info(f"Valeurs manquantes dans {self.name}.csv")
            logger.debug(self.df.isnull().sum())

            # Count lines before filtering
            before = len(self.df)

            # Drop missing values
            self.df = self.df.dropna()

            # Display size of dataframe before and after filtering
            self.dataframe_size(before,1, name=self.name)

            # Reset index to avoid issues
            self.df = self.df.reset_index(drop=True)

        except Exception as e:
            logger.exception("Erreur dans data_nan")


    # Test if dataframe contains datas
    def is_empty(self):
        """
        Checks if the DataFrame is empty after cleaning

        Returns:
            None
        """

        try:
            logger.debug(f"[INIT.is_empty] Colonnes chargées dans {self.name} : {self.df.columns.tolist()}")
            if self.df.empty:
                logger.warning(f"Le DataFrame {self.name} est vide après nettoyage.")
            else:
                logger.info(f"Le DataFrame {self.name} contient {len(self.df)} lignes")

        except Exception as e:
            logger.exception("Erreur dans is_empty")


    # Remove rows with negative values
    def filter_positive(self, columns):
        """
        Removes rows where the value in specified columns is less than or equal to 0.

        Args:
            columns (List[str]): List of column names to apply the filter on.

        Returns:
            pd.DataFrame: Filtered DataFrame with only positive values in specified columns.
        """

        try:
            for col in columns:
                if col not in self.df.columns:
                    logger.warning(f"Colonne '{col}' absente du DataFrame")
                    continue

                # Count lines before filtering
                before = len(self.df)

                # Remove any rows below 0
                self.df = self.df[self.df[col] > 0]
            
                # Display size of dataframe before and after filtering
                self.dataframe_size(before,2, name=self.name, col=col)

        except Exception as e:
            logger.exception("Erreur dans filter_positive")

        return self.df


    # Drop unmatching values
    def drop_invalid_ids(self, column, valid_ids):
        """
        Removes rows with invalid IDs based on a list of valid values.

        Args:
            column (str): Column name containing IDs
            valid_ids (List[Any]): List of valid ID values

        Returns:
            pd.DataFrame: DataFrame containing only rows with valid IDs
        """

        try:
            # Create a dataframe with selected values
            logger.debug("Nom du dataframe : " + self.name + " Les colonnes sont les suivantes " + self.df.columns)
            valid_df = self.df[self.df[column].isin(valid_ids)]

            # Count lines before filtering
            before = len(self.df)
            # Display size of dataframe before and after filtering
            self.dataframe_size(before,3, name=self.name, col=column)


            logger.debug(self.df[~self.df[column].isin(valid_ids)][column].unique())

            # replace original dataframe with cleaned version
            self.df = valid_df

        except KeyError as e:
            logger.exception("Erreur dans drop_invalid_ids")

        return self.df


    # Merge two dataframes
    def merge_count(self, dataframe2, join_column, join, group, column_name , nb_elements, columns_to_merge=None):
        """
        Merge an external dataset with an internal dataset on a specified column,
        then group by the join key, count occurrences, and return the top results.

        Args:
            dataframe2 (pd.DataFrame): The DataFrame to merge
            join_column (str): Column name on which to perform the join.
            join (str): Type of join ('left', 'right', 'inner', 'outer').
            nb_elements (int): Number of top rows to return after sorting.
            group (str or list): Column(s) to group by.
            column_name (str): Name of the count column after grouping.
            columns_to_merge (list or None): List of columns to keep from self.df during merge. If None, keep all.

         Returns:
            pd.DataFrame: Grouped and sorted DataFrame with counts
        """

        try:
            if columns_to_merge is not None:
                df_merge = self.df[columns_to_merge]
            else:
                df_merge = self.df

            merged = dataframe2.merge(df_merge, on=join_column, how=join)
            grouped = merged.groupby(group).size().reset_index(name=column_name)
            return grouped.sort_values(by=column_name, ascending=False).head(nb_elements)

        except Exception as e:
            logger.exception("Erreur dans merge_count")
            return pd.DataFrame()


    # Sort dataframe by a value
    def sort(self, value_sort, nb_elements, dataframe=None, asc=False):
        """
        Sort a dataset by a specified column and return the top rows.

        Args:
            value_sort (str): Name of the column to sort by.
            nb_elements (int): Number of top rows to return.
            dataframe (pd.DataFrame, optional): DataFrame to sort. If None, uses self.df.
            asc (bool, optional): If True, sort in ascending order; otherwise descending (default is False).

        Returns:
            pd.DataFrame: Sorted DataFrame limited to top `nb_elements` rows.
        """

        try:
            if dataframe is None:
                dataframe = self.df
            return dataframe.sort_values(by=value_sort, ascending=asc).head(nb_elements)

        except Exception as e:
            logger.exception("Erreur dans sort")
            return pd.DataFrame()


    # Create a dictionary
    @staticmethod
    def create_dic(data, x, y, title, axis_name_x='x-axis', axis_name_y='y-axis', palette='Blues_d'):
        """
        Create a dictionary containing all parameters needed for plotting a graph.

        Args:
            data (pd.DataFrame): DataFrame to be used for plotting.
            x (str): Column name to be used on the x-axis.
            y (str): Column name to be used on the y-axis.
            title (str): Title of the graph.
            axis_name_x (str) : Title of x-axis
            axis_name_y (str) : Title of y-axis
            palette (str, optional): Color palette for the plot (default is 'Blues_d').

        Returns:
            dict: Dictionary containing the plotting configuration.
        """

        try:
            return {
                'data' : data,
                'x' : x,
                'y' : y,
                'title' : title,
                'axis_name_x' : axis_name_x,
                'axis_name_y' : axis_name_y,
                'palette' : palette,
            }

        except Exception as e:
            logger.exception("Erreur dans create_dic")
            return {}


    def format_message(self, msg_id, **kwargs):
        """
        Generates a formatted log message based on a message ID and optional context.

        Parameters:
            msg_id (int): Identifier of the message type to format.
            **kwargs: Optional keyword arguments to customize the message.
                - name (str): Name of the DataFrame or context (default: 'le dataframe').
                - col (str): Name of the relevant column (if applicable).

        Returns:
            str: The formatted message string corresponding to the message ID.
        """

        match msg_id:
            case self.MSG_MISSING_VALUES:
                return f"Lignes supprimées pour valeurs manquantes dans {kwargs.get('name', 'le dataframe')}\n"
            case self.MSG_INVALID_VALUES:
                return f"Lignes supprimées dans {kwargs.get('name', 'le dataframe')} : {kwargs.get('col', 'colonne non renseigné')} a des valeurs négatives\n"
            case self.MSG_NO_MATCHING_IDS:
                return f"Lignes supprimées : valeurs sans correspondance dans '{kwargs.get('col', 'colonne non renseigné')}' du DataFrame {kwargs.get('name', 'inconnu')}\n"
            case self.MSG_DUPLICATES_VALUES:
                return f"Lignes dupliquées ont été supprimées dans {kwargs.get('name', 'le dataframe')}\n"
            case _:
                return "Message inconnu\n"


    def dataframe_size(self, before, msg_id, **kwargs):
        """
        Logs the size of the DataFrame before and after a filtering operation.

        Parameters:
            before (int): Number of rows in the DataFrame before the operation.
            msg_id (int): Identifier for the type of filtering that was performed.
            **kwargs: Optional keyword arguments to customize the log message.
                - name (str): Name of the DataFrame or context (default: 'le dataframe').
                - col (str): Name of the relevant column (if applicable).

        Behavior:
            - Logs the number of rows before filtering.
            - If any rows were removed, logs a formatted message explaining why.
        """

        # Count how many lines are removed
        removed = before - len(self.df)
        logger.info(f"{before} lignes dans le Dataframe {kwargs.get('name', 'le dataframe')}\n")

        if removed > 0:
            message = self.format_message(msg_id, **kwargs)
            logger.info(f"{removed} {message}")
