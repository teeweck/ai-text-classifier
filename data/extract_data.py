import kagglehub
from pathlib import Path

dataset_dir = "raw"
dataset_id = "abdelmalekeladjelet/sentiment-analysis-dataset"

# Get the directory of the current script
script_directory = Path(__file__).resolve().parent
download_dir = script_directory / dataset_dir

path = kagglehub.dataset_download(dataset_id, output_dir=str(download_dir), force_download=True)

print("Path to downloaded dataset files:", path)
print("Files are located in:", str(download_dir))
