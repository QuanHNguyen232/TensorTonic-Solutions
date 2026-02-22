import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    a, b = np.array(a), np.array(b)
    nume = np.dot(a, b)
    deno = np.linalg.norm(a) * np.linalg.norm(b)

    return nume/deno if deno !=0 else 0