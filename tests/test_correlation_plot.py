import numpy as np
import pandas as pd
import pytest
import altair as alt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.correlation_plot import corr_plot

# Case: should throw an error when feature not exist in dataframe
def test_feature_not_exist():
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "income": [0, 1, 0],})

    feature_list = ["age", "hours_per_week"] 

    with pytest.raises(ValueError, match="Features not found"):
        corr_plot(df, feature_list)

        
# Case: should throw an error when incorrect types are passed to the feature_list argument
def test_feature_list_wrong_type():
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "income": [0, 1, 0],
    })

    feature_list = "age"  

    with pytest.raises(TypeError, match="feature_list must be a list"):
        corr_plot(df, feature_list)

        
# Case: should throw an error when incorrect types are passed to the dataframe argument
def test_dataframe_wrong_type():
    df = {
        "age": [20, 30, 40],
        "income": [0, 1, 0],}  

    feature_list = ["age", "income"]

    with pytest.raises(TypeError, match="dataframe must be a pandas DataFrame"):
        corr_plot(df, feature_list)
