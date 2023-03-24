import pandas as pd
import torch


class ClassificationModelTester:
    def __init__(self, model_path, all_labels, device):
        self.device = device
        self.model = torch.load(model_path).to(self.device)
        self.all_labels = all_labels
        self.id2label = dict(enumerate(sorted(self.all_labels)))
        self.label2id = {v: k for k, v in self.id2label.items()}

    def predict(self, vectors, threshold=0.1):
        probs = torch.sigmoid(
            self.model(torch.from_numpy(vectors.values).float().to(self.device))
        )
        sorted_probs, indices = torch.sort(probs, descending=True)
        results = {}
        for i in range(len(vectors)):
            pred = indices[i][sorted_probs[i] > threshold].cpu().numpy()
            results[vectors.index[i]] = [self.id2label[p] for p in pred]
        return results

    def predict_proba(self, vectors):
        probs = torch.sigmoid(
            self.model(torch.from_numpy(vectors.values).float().to(self.device))
        )
        return pd.DataFrame(
            probs.cpu().detach().numpy(),
            columns=sorted(self.all_labels),
            index=vectors.index,
        )


class SimilarityMatchingModelTester:
    def __init__(self, model_path, all_labels, device):
        self.device = device
        self.model = torch.load(model_path).to(self.device)
        self.model.target_embeddings = self.model.target_embeddings.to(self.device)
        self.all_labels = all_labels
        self.id2label = dict(enumerate(sorted(self.all_labels)))
        self.label2id = {v: k for k, v in self.id2label.items()}

    def predict(self, vectors, num_preds=3):
        cos_sim = self.model(torch.from_numpy(vectors.values).float().to(self.device))
        preds = cos_sim.detach().cpu().numpy().argsort(axis=1)[:, -num_preds:]
        results = {}
        for i in range(len(vectors)):
            pred = preds[i][::-1]
            results[vectors.index[i]] = [self.id2label[p] for p in pred]
        return results
