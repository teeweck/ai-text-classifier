import csv
import kagglehub
from pathlib import Path
import pandas as pd

dataset_dir = "raw"
dataset_id = "abdelmalekeladjelet/sentiment-analysis-dataset"
dataset_path = "data/raw/sentiment_data.csv"

# Get the directory of the current script
script_directory = Path(__file__).resolve().parent
download_dir = script_directory / dataset_dir

path = kagglehub.dataset_download(dataset_id, output_dir=str(download_dir), force_download=True)

print("Path to downloaded dataset files:", path)
print("Files are located in:", str(download_dir))

#---------- Dataset cleaning ----------
df = pd.read_csv(dataset_path)
print(df.head())

# Define your new headers as a list
new_headers = ['Index', 'Comment', 'Sentiment']

# Read the CSV file, overriding existing headers or adding new ones if they don't exist
df = pd.read_csv(dataset_path, header=0, names=new_headers)

# Remove rows with NaN value
df = df.dropna()

print(df.head())

df.to_csv(dataset_path, index=False)