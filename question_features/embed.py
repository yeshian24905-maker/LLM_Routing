"""Sentence embeddings of question text and CLIP embeddings of question images."""

from pathlib import Path

import numpy as np

from .handcrafted import question_text


TEXT_MODEL = "BAAI/bge-base-en-v1.5"
IMAGE_MODEL = "clip-ViT-B-32"
# CLIP resizes to 224 px; shrinking first keeps memory low for large scans.
IMAGE_SIDE = 448


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def embed_text(questions, model_name=TEXT_MODEL):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=_device())
    return model.encode([question_text(q) for q in questions], batch_size=32,
                        normalize_embeddings=True, show_progress_bar=True)


def embed_images(questions, data_dir, model_name=IMAGE_MODEL, batch_size=32):
    """Mean of the normalized CLIP embeddings of each question's images; zeros without images."""
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=_device())
    owners = [(i, path) for i, q in enumerate(questions) for path in q["images"]]
    vectors = []
    for start in range(0, len(owners), batch_size):
        batch = []
        for _, path in owners[start:start + batch_size]:
            with Image.open(Path(data_dir) / path) as image:
                image = image.convert("RGB")
                image.thumbnail((IMAGE_SIDE, IMAGE_SIDE))
                batch.append(image)
        vectors.append(model.encode(batch, normalize_embeddings=True))
        print(f"images {min(start + batch_size, len(owners))}/{len(owners)}", flush=True)
    vectors = np.concatenate(vectors)
    result = np.zeros((len(questions), vectors.shape[1]), dtype=np.float32)
    for (owner, _), vector in zip(owners, vectors):
        result[owner] += vector
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return np.divide(result, norms, out=result, where=norms > 0)
