#!/usr/bin/env python
# clean_split_data.py
# Script 2: clean and split the Adult dataset
# date: 2025-12-04

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


@click.command()
@click.option(
    "--input-path",
    type=str,
    required=True,
    help="Path to the raw Adult data file (e.g., data/raw/adult.data).",
)
@click.option(
    "--output-train-path",
    type=str,
    required=True,
    help="File path to write the cleaned training data CSV.",
)
@click.option(
    "--output-test-path",
    type=str,
    required=True,
    help="File path to write the cleaned test data CSV.",
)
@click.option(
    "--test-size",
    type=float,
    default=0.6,
    show_default=True,
    help="Proportion of data to allocate to the test set.",
)
@click.option(
    "--random-state",
    type=int,
    default=123,
    show_default=True,
    help="Random seed for reproducible splitting.",
)
@click.option(
    "--target-col",
    type=str,
    default="income",
    show_default=True,
    help="Name of the target column.",
)
def main(
    input_path: str,
    output_train_path: str,
    output_test_path: str,
    test_size: float,
    random_state: int,
    target_col: str,
) -> None:
    """Clean the Adult dataset and perform a train/test split."""

    # Load raw data; Adult .data has no header row
    df = pd.read_csv(
        input_path,
        header=None,
        names=COLUMN_NAMES,
        na_values=["?"],
        skipinitialspace=True,
    )
    df = df.replace("?", np.nan)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # Standardize target labels (e.g., '<=50K.' -> '<=50K')
    if df[target_col].dtype == "object":
        df[target_col] = (
            df[target_col]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.strip()
        )

    # Drop duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Train/test split (no stratification, consistent with report)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    # Write outputs
    train_df.to_csv(output_train_path, index=False)
    test_df.to_csv(output_test_path, index=False)

    print(f"Train data saved to: {output_train_path}")
    print(f"Test data saved to: {output_test_path}")


if __name__ == "__main__":
    main()
