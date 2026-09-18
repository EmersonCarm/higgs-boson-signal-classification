"""
Downloads the ATLAS Higgs Boson dataset from the UCI Machine Learning Repository and saves it as a CSV file
and puts it in the data folder. If the data folder does not exist, it will be created, otherwise the script 
does nothing.

Source: ATLAS Collaboration (2014). Dataset from the ATLAS Higgs Boson
Machine Learning Challenge 2014. CERN Open Data Portal.
DOI:10.7483/OPENDATA.ATLAS.ZBP2.M5T8
Record: https://opendata.cern.ch/record/328
"""

import gzip
import shutil
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_PATH = DATA_DIR / "atlas-higgs-challenge-2014-v2.csv"
GZ_PATH = DATA_DIR / "atlas-higgs-challenge-2014-v2.csv.gz" 
DOWNLOAD_URL = "https://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz"

def download_dataset():
    if CSV_PATH.exists():
        print(f"Dataset already exists at {CSV_PATH}. Skipping download.")
        return
    
     else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset from {DOWNLOAD_URL}...")
        response = requests.get(DOWNLOAD_URL, stream=True, timeout=60)
        response.raise_for_status()  # Raise an error for bad responses

        with open(GZ_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print("Download complete. Extracting the dataset...")
        with gzip.open(GZ_PATH, "rb") as f_in:
            with open(CSV_PATH, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        GZ_PATH.unlink()  # Remove the .gz file after extraction
        print(f"Dataset extracted and saved to {CSV_PATH}.")

if __name__ == "__main__":
    download_dataset()