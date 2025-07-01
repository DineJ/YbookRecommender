from data_processing import DatasetCleanerGeneric, DatasetCleanerSpecific
import pandas as pd
import os

# Chargement et nettoyage du fichier books.csv
cleaner = DatasetCleanerSpecific(name="books")  # attend books.csv dans ../datas/raw/

books_clean = cleaner.clean_books(cleaner.df)
print("Books nettoyés :", books_clean.shape)

# Si tu as les autres datasets nettoyés, tu peux les passer à save_clean_data,
# sinon tu peux passer des DataFrames vides temporairement par exemple:
import pandas as pd
empty_df = pd.DataFrame()

# Sauvegarde des fichiers nettoyés (ici on passe des DataFrames vides pour les autres)
cleaner.save_clean_data(
    books_clean=books_clean,
    ratings_clean=empty_df,
    book_tags_clean=empty_df,
    tags_clean=empty_df,
    to_read_clean=empty_df
)

# Affichage des 5 premières lignes du fichier books_clean.csv sauvegardé
output_dir = '../datas/processed'
clean_file = os.path.join(output_dir, 'books_clean.csv')

print("\nAffichage des 5 premières lignes du fichier nettoyé final books_clean.csv :")
df_clean = pd.read_csv(clean_file)
print(df_clean.head())
