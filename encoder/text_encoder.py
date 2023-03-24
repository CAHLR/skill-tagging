import json

import pandas as pd
import sklearn
from sentence_transformers import SentenceTransformer

from . import util


class TextEncoder:
    def __init__(self, device):
        self.device = device
        self.model = SentenceTransformer("paraphrase-mpnet-base-v2").to(self.device)

    def encode(self, path):
        text = json.load(open(path))
        names = list(text.keys())
        contents = [text[name] for name in names]
        vectors = self.model.encode(contents)
        vectors = util.l2_normalize(vectors)
        vectors = pd.DataFrame(vectors, index=names)
        return vectors
