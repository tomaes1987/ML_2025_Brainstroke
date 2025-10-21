import os
import pandas as pd
import warnings
import ssl
import requests
from io import StringIO

warnings.filterwarnings('ignore')
ssl._create_default_https_context = ssl._create_unverified_context
pd.set_option('display.max_columns', None)

def load_and_save_csv(url=None, save_folder='dataset/raw'):
    """
    Loads CSV from a link and saves it to 'dataset/raw' folder.
    If URL is not provided, asks the user to paste it.

    Args:
        url (str, optional): The link to CSV file. Default is None.
        save_folder (str): Folder to save CSV (default 'dataset/raw').

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    # Ensure the folder exists
    os.makedirs(save_folder, exist_ok=True) #os.makedirs is responsible for creating directories, os stands for operating system, exist_ok=True prevents error if directory exists

    # List existing files in the folder
    existing_files = [f for f in os.listdir(save_folder) if os.path.isfile(os.path.join(save_folder, f))] #isfile checks if the path is a file, .join combines folder path and file name
    if existing_files:
        print("Existing files in the folder:")
        for file in existing_files:
            print(f"- {file}")
    else:
        print(f"No files found in the folder '{save_folder}'.")

    # Get a valid URL from the user
    while not url or not url.lower().endswith('.csv'):
        url = input("Provide URL (must end with .csv) or type 'e' to exit: ").strip()
        if url.lower() == 'e':
            print("Exiting without downloading.")
            df = pd.read_csv(os.path.join(save_folder, existing_files[0])) if existing_files else None
            print (df.head())
            return df

    # Extract the filename and prepare the save path
    filename = url.split("/")[-1]
    save_path = os.path.join(save_folder, filename)

    # Check if the file already exists
    if os.path.exists(save_path):
        print(f"File '{filename}' already exists. Loading the dataset...")
        try:
            existing_df = pd.read_csv(save_path)
            print("Preview of the existing dataset:")
            print(existing_df.head())
            return existing_df
        except Exception as e:
            print(f"Error reading the existing file: {e}")
            return None

    # Download and save the file
    print(f"Downloading {url}...")
    try:
        response = requests.get(url, verify=False)
        response.raise_for_status()  # Ensure the request was successful
        df = pd.read_csv(StringIO(response.text))
        df.to_csv(save_path, index=False)
        print(f"Dataframe saved in {save_path}")
        print("Preview of the dataframe:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error during download or saving: {e}")
        return None
    