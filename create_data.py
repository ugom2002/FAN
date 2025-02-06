# create_data.py
import os
import urllib.request

# Créer le dossier 'data' s'il n'existe pas
data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Dossier '{data_dir}' créé avec succès.")

# URL du fichier ETTh1.csv depuis GitHub
url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
file_path = os.path.join(data_dir, "ETTh1.csv")

# Télécharger le fichier
try:
    urllib.request.urlretrieve(url, file_path)
    print(f"Fichier téléchargé avec succès : {file_path}")
except Exception as e:
    print(f"Erreur lors du téléchargement : {e}")
