# create_data.py
import os
import urllib.request

# Create the 'data' directory if it doesn't exist
data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Directory '{data_dir}' created successfully.")

# URL of the ETTh1.csv file from GitHub
url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
file_path = os.path.join(data_dir, "ETTh1.csv")

# Download the file
try:
    urllib.request.urlretrieve(url, file_path)
    print(f"File downloaded successfully: {file_path}")
except Exception as e:
    print(f"Error while downloading the file: {e}")
