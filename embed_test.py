from fastembed import TextEmbedding
import numpy as np

model = TextEmbedding("BAAI/bge-small-en-v1.5")

texts = [
    "I study cybersecurity at university.",
    "My favourite food is jollof rice.",
    "I am learning to build AI agents.",
]
question = "What are you learning about AI?"

vecs = np.array(list(model.embed(texts)))
q = np.array(list(model.embed([question])))[0]

# cosine similarity: higher = closer in meaning
scores = vecs @ q / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(q))

for text, score in zip(texts, scores):
    print(f"{score:.3f}  {text}")