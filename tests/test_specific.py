import pandas as pd
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from src.data_processing import DatasetCleanerSpecific

@pytest.fixture
def cleaner():
    return DatasetCleanerSpecific(name="test", path="tests/data/")

def test_clean_books(cleaner):
    df = pd.DataFrame({
        'book_id': [1, -1],
        'best_book_id': [1, 1],
        'work_id': [1, 1],
        'books_count': [10, 5],
        'isbn13': [9781234567890, 1234567890123],
        'isbn': ['abc', 'def'],
        'small_image_url': ['url1', 'url2'],
        'work_ratings_count': [1, 2],
        'ratings_count': [3, 4]
    })
    result = cleaner.clean_books(df)
    assert len(result) == 1
    assert 'isbn' not in result.columns
    assert result.iloc[0]['isbn13'] == 9781234567890

def test_clean_tags(cleaner):
    df = pd.DataFrame({
        'tag_id': [1, -1],
        'tag_name': ['-sci-fi-', ' fantasy  ']
    })
    result = cleaner.clean_tags(df)
    assert len(result) == 1
    assert result.iloc[0]['tag_name'] == 'sci fi'

def test_clean_book_tag(cleaner):
    tags = pd.DataFrame({'tag_id': [100]})
    books = pd.DataFrame({'book_id': [200]})
    df = pd.DataFrame({
        'tag_id': [100, 999],
        'book_id': [200, 999],
        'count': [1, 2],
        'goodreads_book_id': [123, 456]
    })
    result = cleaner.clean_book_tag(df, tags, books)
    assert len(result) == 1
    assert result.iloc[0]['tag_id'] == 100

def test_clean_ratings(cleaner):
    books = pd.DataFrame({'book_id': [10]})
    df = pd.DataFrame({
        'user_id': [1, -1],
        'book_id': [10, 999],
        'rating': [5, 0]
    })
    result = cleaner.clean_ratings(df, books)
    assert len(result) == 1
    assert result.iloc[0]['rating'] == 5

def test_clean_to_read(cleaner):
    books = pd.DataFrame({'book_id': [101]})
    df = pd.DataFrame({
        'user_id': [1, 2, 3],
        'book_id': [101, 999, -5]
    })
    result = cleaner.clean_to_read(df, books)
    assert len(result) == 1
    assert result.iloc[0]['book_id'] == 101
