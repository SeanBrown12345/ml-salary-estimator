# data_download.py
# Modified From breast-cancer-predictor ttimbers, https://github.com/ttimbers/breast-cancer-predictor/blob/2.0.0/scripts/download_data.py
# date: 2025-12-03


import click
import os
import zipfile
import requests

def read_zip(url, destination_path):
    """
    Read a zip file from the given URL and extract its contents to the specified directory.

    Parameters:
    ----------
    url : str
        The URL of the zip file to be read.
    directory : str
        The directory where the contents of the zip file will be extracted.

    Returns:
    -------
    None
    """
    
    request = requests.get(url)
    filename_from_url = os.path.basename(url)

    if request.status_code != 200:
        raise ValueError('The URL is invalid.')

    if filename_from_url[-4:] != '.zip':
        raise ValueError('The URL provided does not point to a zip file.')

    # Save the zip file to the destination path
    path_to_zip_file = os.path.join(destination_path, filename_from_url)
    with open(path_to_zip_file, 'wb') as f:
        f.write(request.content)
        

   
    original_files = os.listdir(destination_path)
    original_timestamps = []
    for filename in original_files:
        filename = os.path.join(destination_path, filename)
        original_timestamp = os.path.getmtime(filename)
        original_timestamps.append(original_timestamp)

    # Extract the contents of the zip file
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(destination_path)

    # Check if the zip file was empty
    current_files = os.listdir(destination_path)
    current_timestamps = []
    for filename in current_files:
        filename = os.path.join(destination_path, filename)
        current_timestamp = os.path.getmtime(filename)
        current_timestamps.append(current_timestamp)
    if (len(current_files) == len(original_files)) & (original_timestamps == current_timestamps):
        raise ValueError('The ZIP file is empty.')
    

@click.command()
@click.option('--url', type=str, help="URL of dataset to be downloaded")
@click.option('--destination_path', type=str, help="Path to directory where raw data will be written to")

def main(url, destination_path):
    try:
        read_zip(url, destination_path)
    except:
        os.makedirs(destination_path)
        read_zip(url, destination_path)

if __name__ == "__main__":
    main() 