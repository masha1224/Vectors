import os
import torch
import joblib
from PIL import Image
from itertools import islice
from sentence_transformers import SentenceTransformer
from tqdm.notebook import tqdm
from sqlalchemy.orm import Session
from .sql_alchemy_model import Img, engine

# Parameters
MAX_IMAGES = 100
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
            
            # Calculate embeddings
            embeddings = model.encode(images, batch_size=BATCH_SIZE, device=device, convert_to_tensor=False)

            # Convert embeddings to string (for now — change to pgvector if you switch types)
            img_objects = [
                Img(image_path=path, embedding=",".join(map(str, embedding)))
                for path, embedding in zip(image_paths_batch, embeddings)
            ]

            # Insert batch
            insert_images(engine, img_objects)

            # Update progress
            pbar.update(len(img_objects))

# Example use
# image_paths = [...]  # List of paths to images
# vectorize_images(engine, model, image_paths)
