import os
import pathlib

import pandas as pd
import torch

from . import util
from .pytorch_compact_bilinear_pooling import compact_bilinear_pooling


class Fusion:
    def __init__(self):
        self.model = compact_bilinear_pooling.CompactBilinearPooling(768, 2560, 4096)
        self.model.load_state_dict(
            torch.load(
                os.path.join(
                    pathlib.Path(__file__).parent, "compact_bilinear_pooling.ckpt"
                )
            )
        )

    def fuse(self, text_embeddings, image_embeddings):
        names = text_embeddings.index.tolist()
        image_embeddings = image_embeddings.loc[names, :]
        fused_vectors = self.model(
            torch.from_numpy(text_embeddings.values),
            torch.from_numpy(image_embeddings.values),
        )
        fused_vectors = util.l2_normalize(fused_vectors.numpy())
        return pd.DataFrame(fused_vectors, index=names)
