import pandas as pd
import os
import pytest
import pandera as pa
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.validate_data import validate_data


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


# Case： should pass validation when the dataframe fully satisfies the schema
def test_validate_data_passes(sample_data):
    validated_df = validate_data(sample_data)
    assert isinstance(validated_df, pd.DataFrame)
    assert validated_df.shape == sample_data.shape

# Case: should raise a TypeError when the input is not a pandas DataFrame
def test_validate_data_not_dataframe():
    invalid_input = {
        "age": [30, 40],
        "income": ["<=50K", ">50K"]}

    with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
        validate_data(invalid_input)

# Case: should raise a ValueError when the dataframe is empty
def test_validate_data_empty_dataframe():
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="Dataframe must contain observations"):
        validate_data(empty_df)
        

# Case: should raise an error when the dataframe contains duplicate rows
def test_validate_data_duplicate_rows(sample_data):
    df_dup = pd.concat([sample_data, sample_data.iloc[[0]]], ignore_index=True)

    with pytest.raises(pa.errors.SchemaError, match="Duplicate rows found"):
        validate_data(df_dup)
        
# Case: should raise an error when the income column contains a value outside the allowed set ("<=50K", ">50K")
def test_validate_data_invalid_income(sample_data):
    df_invalid = sample_data.copy()
    df_invalid.loc[0, "income"] = "50K+"  # invalid category

    with pytest.raises(pa.errors.SchemaError):
        validate_data(df_invalid)


        
# Case: should raise an error when a required column defined in the schema
def test_validate_data_missing_column(sample_data):
    df_missing = sample_data.drop(columns=["age"])

    with pytest.raises(pa.errors.SchemaError):
        validate_data(df_missing)