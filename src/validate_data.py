import pandera.pandas as pa
import pandas as pd

def validate_data(dataframe):
    """
    Validate the input dataframe for type, structure, and data integrity.

    This function checks that the columns in the input DataFrame conform to the expected types and value ranges.
    It also ensures there are no duplicate rows and no entirely empty rows.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input dataframe containing the raw observations to be validated.

    Returns
    -------
    pandas.DataFrame
        The validated dataframe. If all checks pass, the returned dataframe is identical to the input.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.
    ValueError
        If the dataframe is empty and contains no observations.
    pandera.errors.SchemaError
        If the dataframe violates any column-level or dataframe-level constraints defined in the Pandera schema.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")    
    if dataframe.empty:
        raise ValueError("Dataframe must contain observations.")
    
    schema = pa.DataFrameSchema(
        {
            "age": pa.Column(pa.Int, pa.Check.between(17, 90)),
            "workclass": pa.Column(str, nullable=True),
            "fnlwgt": pa.Column(pa.Int),
            "education": pa.Column(str),
            "education_num": pa.Column(pa.Int, pa.Check.ge(1)),
            "marital_status": pa.Column(str),
            "occupation": pa.Column(str, nullable=True),
            "relationship": pa.Column(str),
            "race": pa.Column(str),
            "sex": pa.Column(str, pa.Check.isin(["Male", "Female"])),
            "capital_gain": pa.Column(pa.Int, pa.Check.ge(0)),
            "capital_loss": pa.Column(pa.Int, pa.Check.ge(0)),
            "hours_per_week": pa.Column(pa.Int, pa.Check.between(1, 99)),
            "native_country": pa.Column(str, nullable=True),
            "income": pa.Column(str, pa.Check.isin(["<=50K", ">50K"])),
        },
        checks=[
            pa.Check(
                lambda df: ~(df.isna().all(axis=1)).any(),error="Empty rows found."),
            pa.Check(lambda df: ~df.duplicated().any(),error="Duplicate rows found.")]    )

    return schema.validate(dataframe, lazy=False)

