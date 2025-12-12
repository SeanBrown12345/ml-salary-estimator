import os
import pytest
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.load_data import load_data

def test_load_success(tmp_path):
    csv = tmp_path/'data.csv'
    csv.write_text('col1, col2, col3\n1,2,3\n4,5,6')
    res = load_data(csv, csv)
    
    assert res != 1
    
    train, test = res
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert train.shape == (2, 3)
    assert train.equals(test)
    
def test_load_fail(tmp_path):
    res = load_data(tmp_path/'nofile.csv', tmp_path/'nopath2.csv')
    assert res == 1
