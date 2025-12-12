import pandas as pd

def load_data(train_file, test_file):
    """
    Loads training and testing datasets from CSV files

    Args:
        train_file: The file path for the training data CSV
        test_file: The file path for the test data CSV

    Returns:
        tuple(pd.DataFrame, pd.DataFrame) | int: 
        A tuple containing (train_df, test_df) if the file reading is successful 
        Returns 1 if an exception occurs during file reading
    """
    try:
        train_df = pd.read_csv(train_file)
        print('Successfully loaded training data')
        test_df = pd.read_csv(test_file)
        print('Successfully loaded test data')
        return(train_df, test_df)
    except Exception as e:
        print(f'Error when reading files: {e}')
        return(1)
