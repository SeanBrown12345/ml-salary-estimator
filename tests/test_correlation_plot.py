import numpy as np
import pandas as pd
import pandera as pa
import pytest
import altair as alt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.correlation_plot import corr_plot

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "age": [40, 42, 57, 49, 64, 26, 42, 63, 22, 42, 35, 38, 19],
        "workclass": [
            "Private", "Private", "Private", "Private", "Self-emp-not-inc",
            "Private", "Private", None, "Private", "Private",
            "Private", "Private", "Private"
        ],
        "fnlwgt": [
            137142, 251239, 365683, 93639, 159938, 133766, 111483,
            234083, 181773, 193626, 31269, 283122, 376683
        ],
        "education": [
            "HS-grad", "Some-college", "HS-grad", "Bachelors", "HS-grad",
            "HS-grad", "Bachelors", "HS-grad", "HS-grad", "HS-grad",
            "HS-grad", "HS-grad", "Some-college"
        ],
        "education_num": [9, 10, 9, 13, 9, 9, 13, 9, 9, 9, 9, 9, 10],
        "marital_status": [
            "Married-civ-spouse", "Married-civ-spouse", "Divorced",
            "Married-civ-spouse", "Married-civ-spouse", "Married-civ-spouse",
            "Married-civ-spouse", "Divorced", "Never-married",
            "Married-civ-spouse", "Married-civ-spouse",
            "Married-civ-spouse", "Never-married"
        ],
        "occupation": [
            "Sales", "Exec-managerial", "Machine-op-inspct",
            "Exec-managerial", "Craft-repair", "Farming-fishing",
            "Tech-support", None, "Transport-moving", "Adm-clerical",
            "Exec-managerial", "Machine-op-inspct", "Other-service"
        ],
        "relationship": [
            "Husband", "Husband", "Not-in-family", "Wife", "Husband",
            "Husband", "Husband", "Not-in-family", "Own-child", "Wife",
            "Husband", "Husband", "Unmarried"
        ],
        "race": [
            "White", "White", "White", "White", "White", "White",
            "White", "White", "Black", "White", "White", "White", "Black"
        ],
        "sex": [
            "Male", "Male", "Female", "Female", "Male", "Male",
            "Male", "Female", "Male", "Female", "Male", "Male", "Female"
        ],
        "capital_gain": [0, 0, 0, 0, 2635, 0, 0, 0, 0, 0, 0, 0, 2036],
        "capital_loss": [0, 0, 0, 0, 0, 0, 0, 2205, 0, 0, 0, 0, 0],
        "hours_per_week": [40, 40, 40, 43, 24, 70, 50, 40, 40, 53, 40, 40, 30],
        "native_country": [
            "United-States", "Puerto-Rico", "United-States", "United-States",
            "Italy", "United-States", "United-States", "United-States",
            "United-States", "United-States", "United-States",
            "United-States", "United-States"
        ],
        "income": [
            ">50K", "<=50K", ">50K", "<=50K", "<=50K", "<=50K", ">50K",
            "<=50K", "<=50K", "<=50K", "<=50K", ">50K", "<=50K"
        ],
    })
    
# Case: should throw an error when feature not exist in dataframe
def test_feature_not_exist(sample_data):
    feature_list = ["age", "non_existing_feature"]

    with pytest.raises(ValueError, match="Features not found"):
        corr_plot(sample_data, feature_list)

        
# Case: should throw an error when incorrect types are passed to the feature_list argument
def test_feature_list_wrong_type(sample_data):
    feature_list = "age"  

    with pytest.raises(TypeError, match="feature_list must be a list"):
        corr_plot(sample_data, feature_list)

        
# Case: should throw an error when incorrect types are passed to the dataframe argument
def test_dataframe_wrong_type():
    df = {
        "age": [20, 30, 40],
        "income": [0, 1, 0],}  

    feature_list = ["age", "income"]

    with pytest.raises(TypeError, match="dataframe must be a pandas DataFrame"):
        corr_plot(df, feature_list)

# Case: should throw an error when feature_list is empty
def test_feature_list_empty(sample_data):
    feature_list = []

    with pytest.raises(ValueError):
        corr_plot(sample_data, feature_list)
        

# Case: should return an Altair Chart when valid inputs are provided
def test_corr_plot_valid_input(sample_data):
    feature_list = ["age", "education_num", "hours_per_week"]

    chart = corr_plot(sample_data, feature_list)

    assert isinstance(chart, alt.Chart)
