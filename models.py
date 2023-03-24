import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationModel(nn.Module):
    def __init__(self, n_dim, n_class):
        super(ClassificationModel, self).__init__()
        self.fc = nn.Linear(n_dim, n_dim)
        self.classifier = nn.Linear(n_dim, n_class)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.classifier(x)
        return x


class SimilarityMatchingModel(nn.Module):
    def __init__(self, dim, target_embeddings):
        super(SimilarityMatchingModel, self).__init__()
        self.dim = dim
        self.target_embeddings = target_embeddings
        self.model = nn.Linear(self.dim, self.dim, bias=False)
        self.cos_sim = nn.CosineSimilarity(dim=2, eps=1e-6)

    def forward(self, x):
        if x.shape[1] < self.dim:
            x = F.pad(x, (0, self.dim - x.shape[1]), "constant", 0)
        translated = self.model(x)
        return self.cos_sim(
            *torch.broadcast_tensors(
                translated.unsqueeze(1), self.target_embeddings.unsqueeze(0)
            )
        ).detach()
