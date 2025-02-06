# create_data.py
import os
import numpy as np
import pandas as pd

# Création du sous-répertoire "data" s'il n'existe pas
data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Création du répertoire '{data_dir}'.")

# Génération de données simulées pour la série temporelle ETTh1
# Par exemple, une sinusoïde bruitée
n_points = 1000  # nombre de points
dates = pd.date_range(start="2020-01-01", periods=n_points, freq="H")
values = np.sin(np.linspace(0, 10 * np.pi, n_points)) + np.random.normal(scale=0.1, size=n_points)

# Création d'un DataFrame avec une colonne "date" et une colonne "value"
df = pd.DataFrame({
    "date": dates,
    "value": values
})

# Sauvegarde du DataFrame dans le fichier CSV
csv_path = os.path.join(data_dir, "ETTh1.csv")
df.to_csv(csv_path, index=False)
print(f"Fichier '{csv_path}' créé avec succès.")
