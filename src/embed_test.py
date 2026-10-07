from numpy import dot
from numpy.linalg import norm


from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "How do I get a refund?",
    "Can I have my money back?",
    "The office is closed on Friday.",
]

vectors = model.encode(texts)
print("Shape:", vectors.shape)
print("First vector (first 5 numbers):", vectors)



def cosine(a, b):
    return dot(a, b) / (norm(a) * norm(b))

print("1 vs 2:", cosine(vectors[0], vectors[1]))
print("1 vs 3:", cosine(vectors[0], vectors[2]))
print("2 vs 3:", cosine(vectors[1], vectors[2]))


