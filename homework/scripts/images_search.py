import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import sys  # Added sys import
from pathlib import Path
import torch
from sqlalchemy.orm import Session
from sqlalchemy import select


# Added sys.path modification to allow finding the 'homework' module
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from homework.sql_alchemy_model import Img, engine  # Changed back to absolute import


class ImageSearch:
    def __init__(self, engine, model):
        self.engine = engine  # Silnik wyszukiwania, który zapytuje bazę danych obrazów
        self.model = model    # Model, który tworzy embeddingi (np. model transformera)

    def __call__(self, image_description: str, k: int):
        # Znajdź najbardziej podobne obrazy na podstawie opisu tekstowego i pokaż je
        found_images = self.find_similar_images(image_description, k)
        self.display_images(found_images)


    def find_similar_images(self, image_description: str, k: int):
        image_embedding = self.get_embedding(image_description)
        # remember about session and commit
        with Session(engine) as session:
            result = session.execute(
                select(Img)
                .order_by(
                    Img.embedding.cosine_distance(image_embedding)
                )
                .limit(k),
                execution_options={"prebuffer_rows": True},
            )
            return result.scalars().all()

    def get_embedding(self, description: str):
        # Użyj modelu do uzyskania embeddingu dla opisu (np. modelu transformera)
        embedding = self.model.encode(description)  # Zastąp to odpowiednią metodą modelu
        return embedding

    def execute_query(self, query: str):
        # Wykonaj zapytanie do bazy danych, aby pobrać embeddingi obrazów (np. używając silnika wyszukiwania)
        # To jest funkcja pomocnicza, która powinna zwrócić wyniki z bazy danych
        result = self.engine.run(query)
        return result

    def display_images(self, images):
        # Wyświetl obrazy w jednym rzędzie
        print(images)
        fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
        
        for i, img in enumerate(images):
            img_path = img.image_path
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].axis("off")
            axes[i].set_title(f"Obraz {i+1}")
        
        plt.show()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = SentenceTransformer("clip-ViT-B-32", device=device)

# Stwórz instancję klasy ImageSearch
image_search = ImageSearch(engine, model)

# Wykonaj wyszukiwanie na podstawie opisu
image_search('lamp', k=5)
