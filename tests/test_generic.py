import pandas as pd
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from src.data_processing import DatasetCleanerGeneric

@pytest.fixture
def cleaner():
    return DatasetCleanerGeneric(name="fake", path="tests/data/")  # mocké avec un DataFrame vide

def test_reset_index(cleaner):
    df = pd.DataFrame({'a': [1, 2]}, index=[5, 6])
    result = cleaner.reset_index(df)
    assert list(result.index) == [0, 1]

def test_filter_positive(cleaner):
    df = pd.DataFrame({
        'book_id': [1, -2, 3],
        'user_id': [0, 1, 2]
    })
    result = cleaner.filter_positive(df, ['book_id', 'user_id'])
    assert len(result) == 1
    assert result.iloc[0]['book_id'] == 3
    assert result.iloc[0]['user_id'] == 2

def test_drop_invalid_ids(cleaner):
    df = pd.DataFrame({'book_id': [1, 2, 3]})
    valid_ids = [1, 3]
    result = cleaner.drop_invalid_ids(df, 'book_id', valid_ids)
    assert list(result['book_id']) == [1, 3]