import pandas as pd
import os
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.baseline_test import baseline_test

def test_baseline_test():
    X = pd.DataFrame({'foo': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    y = pd.Series(['>50K', '<=50K', '<=50K', '<=50K', '>50K',
    '>50K', '<=50K', '<=50K', '<=50K', '<=50K',
    '>50K', '<=50K', '<=50K', '<=50K', '>50K',
    '>50K', '<=50K', '<=50K', '<=50K', '<=50K'])
    dummy = DummyClassifier(strategy='constant', constant='>50K')
    dummy_pipe = make_pipeline(dummy)
    models = {'dummy': dummy_pipe}
    res = baseline_test(models, X, y)
        
    assert isinstance(res, pd.DataFrame)
    assert list(res.columns) == ['Model', 'Mean f1 score']
    assert len(res) == 1
