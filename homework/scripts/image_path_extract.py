import pandas as pd
from pathlib import Path

# Ścieżka do pliku
csv_path = Path("../images/metadata/images.csv.gz")

# Wczytanie danych
df = pd.read_csv(csv_path)

# Filtrowanie
filtered = df[(df["width"] >= 1000) & (df["height"] >= 1000)]

# Wyciąganie ścieżek do obrazów
image_paths = filtered["path"].tolist()

# Zapis do pliku tekstowego
with open("valid_image_paths.txt", "w") as f:
    for path in image_paths:
        f.write(f"{path}\n")

print(f"Znaleziono {len(image_paths)} obrazów >= 1000x1000 px")
