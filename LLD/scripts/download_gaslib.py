"""
download_gaslib.py
──────────────────
Utility to fetch GasLib XML networks directly from the Zuse Institute Berlin (ZIB) 
repository for benchmarking pipeline simulations.
"""
import os
import urllib.request
import zipfile

ZIB_BASE_URL = "http://gaslib.zib.de/GasLib-11/GasLib-11.zip"
DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_gaslib")

def download_gaslib11():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    zip_path = os.path.join(DOWNLOAD_DIR, "GasLib-11.zip")
    
    print(f"Downloading GasLib-11 from {ZIB_BASE_URL}...")
    try:
        urllib.request.urlretrieve(ZIB_BASE_URL, zip_path)
        print("Download complete. Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DOWNLOAD_DIR)
        print(f"Extracted to {DOWNLOAD_DIR}")
    except Exception as e:
        print(f"Failed to download or extract: {e}")
        print("Note: If the download fails due to network restrictions, you can "
              "manually download the dataset from http://gaslib.zib.de/ "
              "and place the .net and .cs files in the data/raw_gaslib/ directory.")

if __name__ == "__main__":
    download_gaslib11()
