import numpy as np


def l2_normalize(vectors):
    return vectors / (np.linalg.norm(vectors, ord=2, axis=1) + 1e-8).reshape(
        vectors.shape[0], 1
    )
