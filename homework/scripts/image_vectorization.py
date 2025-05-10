import os
import sys  # Added sys import
from pathlib import Path

# Added sys.path modification to allow finding the 'homework' module
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import joblib
from PIL import Image
from itertools import islice
from sentence_transformers import SentenceTransformer
from tqdm.notebook import tqdm
from sqlalchemy.orm import Session
from homework.sql_alchemy_model import Img, engine  # Changed back to absolute import

# Parameters
MAX_IMAGES = 1000
BATCH_SIZE = joblib.cpu_count(only_physical_cores=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = SentenceTransformer("clip-ViT-B-32", device=device)

# Helper: chunked batching generator
def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch

# Helper: insert to DB
def insert_images(engine, images):
    with Session(engine) as session:
        session.add_all(images)
        session.commit()

# Vectorization
def vectorize_images(engine, model, image_paths):
    image_paths = image_paths[:MAX_IMAGES]  # limit to MAX_IMAGES
    with tqdm(total=MAX_IMAGES) as pbar:
        for image_paths_batch in batched(image_paths, BATCH_SIZE):
            images = [Image.open(path).convert("RGB") for path in image_paths_batch]
            print(f"Processing {len(images)} images...")        
            # Calculate embeddings
            embeddings = model.encode(images, batch_size=BATCH_SIZE, device=device, convert_to_tensor=False)

    
            img_objects = [
                Img(image_path=path, embedding=embedding)
                for path, embedding in zip(image_paths_batch, embeddings)
            ]

            # Insert batch
            insert_images(engine, img_objects)

            # Update progress
            pbar.update(len(img_objects))

# Example use
base_path = Path("../images/small")
relative_paths = Path("valid_image_paths.txt").read_text().splitlines()
absolute_paths = [str(base_path / p) for p in relative_paths]

training=False
if training:
    # Vectorize images
    vectorize_images(engine, model, absolute_paths) 
