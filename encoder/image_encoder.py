import os

import numpy as np
import pandas as pd
import torch
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import transforms

from . import util


class ImageEncoder:
    class SquarePad:
        def __call__(self, image):
            w, h = image.size
            max_wh = np.max([w, h])
            hp1 = int((max_wh - w) / 2)
            vp1 = int((max_wh - h) / 2)
            hp2 = max_wh - w - hp1
            vp2 = max_wh - h - vp1
            padding = (hp1, vp1, hp2, vp2)
            return transforms.functional.pad(
                image, padding, (255, 255, 255), "constant"
            )

    def __init__(self, device):
        self.device = device
        self.square_pad = ImageEncoder.SquarePad()
        self.model = EfficientNet.from_pretrained("efficientnet-b7").to(self.device)
        self.tfms = transforms.Compose(
            [
                self.square_pad,
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def encode(self, path):
        names = []
        vectors = []
        for file in os.listdir(path):
            filepath = os.path.join(path, file)
            foreground = Image.open(filepath).convert("RGBA")
            background = Image.new("RGBA", foreground.size, (255, 255, 255))
            img = Image.alpha_composite(background, foreground).convert("RGB")
            img = self.tfms(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model._avg_pooling(
                    self.model.extract_features(img)
                ).flatten()
            names.append(os.path.splitext(file)[0])
            vectors.append(output.cpu().numpy())
        vectors = util.l2_normalize(np.array(vectors))
        vectors = pd.DataFrame(vectors, index=names)
        return vectors
