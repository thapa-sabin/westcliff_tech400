import numpy as np

def cosine_similarity(vec_a, vec_b):
    intersection = set(vec_a.keys()) & set(vec_b.keys())
    if not intersection:
        return 0.0

    # Calculate dot product
    dot_product = np.dot(
        np.array([vec_a[term] for term in intersection]),
        np.array([vec_b[term] for term in intersection])
    )

    # Calculate magnitudes
    magnitude_a = np.sqrt(np.sum(np.array([vec_a[term] ** 2 for term in vec_a.keys()])))
    magnitude_b = np.sqrt(np.sum(np.array([vec_b[term] ** 2 for term in vec_b.keys()])))

    return dot_product / (magnitude_a * magnitude_b) if magnitude_a and magnitude_b else 0.0

