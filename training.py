import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import models


class ClassificationModelTrainer:
    """A model trainer for classification model.

    Key arguments:
    embeddings -- pandas.DataFrame, the embeddings of the source contents
    labels -- pandas.DataFrame, the labels of the source contents, should have two columns "source" and "target"
    all_labels -- a list of all target labels
    """

    def __init__(
        self,
        embeddings,
        labels,
        all_labels,
        device,
        learning_rate=1e-4,
        val_size=0.1,
        max_none_decreasing_epochs=100,
        verbose=True,
    ):
        self.embeddings = embeddings
        self.labels = labels
        self.all_labels = all_labels
        self.device = device
        self.learning_rate = learning_rate
        self.val_size = val_size
        self.max_none_decreasing_epochs = max_none_decreasing_epochs
        self.verbose = verbose

        self.label2id, self.id2label, self.labels_multihot = self._process_labels()
        (
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.pos_weight,
        ) = self._process_data()
        self.model = models.ClassificationModel(
            self.embeddings.shape[1], len(self.label2id)
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _process_labels(self):
        id2label = dict(enumerate(sorted(self.all_labels)))
        label2id = {v: k for k, v in id2label.items()}
        labels_multihot = self.labels.groupby("source").apply(
            self._multihot, label2id=label2id
        )
        labels_multihot = pd.DataFrame(
            labels_multihot.tolist(), index=labels_multihot.index
        )
        return label2id, id2label, labels_multihot

    def _multihot(self, label, label2id):
        label_ids = [label2id[l] for l in label["target"].unique()]
        label_multihot = np.zeros(len(label2id))
        label_multihot[label_ids] = 1
        return label_multihot

    def _process_data(self):
        indices = self.labels["source"].unique()
        X = self.embeddings.loc[indices, :].values
        X = torch.from_numpy(X).float().to(self.device)
        y = self.labels_multihot.loc[indices, :].values
        y = torch.from_numpy(y).float().to(self.device)
        permutation = np.random.permutation(len(indices))
        val_indices = permutation[: int(len(indices) * self.val_size)]
        train_indices = permutation[int(len(indices) * self.val_size) :]
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        pos_weight = (torch.sum(1 - y_train, dim=0) + 1) / (
            torch.sum(y_train, dim=0) + 1
        )
        return X_train, y_train, X_val, y_val, pos_weight

    def _train_model_one_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        pred = self.model(self.X_train)
        loss = F.binary_cross_entropy_with_logits(
            pred, self.y_train, pos_weight=self.pos_weight
        )
        loss.backward()
        self.optimizer.step()

    def get_val_loss(self):
        pred = self.model(self.X_val)
        loss = F.binary_cross_entropy_with_logits(pred, self.y_val)
        return loss.item()

    def train(self):
        epoch, none_decreasing_epochs, best_loss = 0, 0, float("inf")
        while none_decreasing_epochs < self.max_none_decreasing_epochs:
            self._train_model_one_epoch()
            val_loss = self.get_val_loss()
            if val_loss < best_loss:
                best_loss = val_loss
                none_decreasing_epochs = 0
            else:
                none_decreasing_epochs += 1
            if self.verbose and epoch % 10 == 0:
                print(f"Epoch={epoch:04d} Validation Loss={val_loss:.5f}")
            epoch += 1

    def save_model(self, path):
        torch.save(self.model, path)

    def load_model(self, path):
        self.model = torch.load(path)


class SimilarityMatchingModelTrainer:
    """A model trainer for similarity matching model.

    Key arguments:
    source_embeddings -- pandas.DataFrame, the embeddings of the source contents
    target_embeddings -- pandas.DataFrame, the embeddings of the target labels
    labels -- pandas.DataFrame, the labels of the source contents, should have two columns "source" and "target"
    all_labels -- a list of all target labels
    """

    def __init__(self, source_embeddings, target_embeddings, labels, all_labels):
        self.all_labels = all_labels
        self.labels = labels
        self.source_embeddings = source_embeddings
        self.target_embeddings = target_embeddings.loc[self.all_labels]
        self.dim = max(source_embeddings.shape[1], target_embeddings.shape[1])

        self._process_data()
        self.model = models.SimilarityMatchingModel(
            self.dim, torch.from_numpy(self.target_embeddings.values).float()
        )

    def _process_data(self):
        if self.target_embeddings.shape[1] < self.dim:
            self.target_embeddings = pd.DataFrame(
                np.pad(
                    self.target_embeddings.values,
                    ((0, 0), (0, self.dim - self.target_embeddings.shape[1])),
                ),
                index=self.target_embeddings.index,
            )
        if self.source_embeddings.shape[1] < self.dim:
            self.source_embeddings = pd.DataFrame(
                np.pad(
                    self.source_embeddings.values,
                    ((0, 0), (0, self.dim - self.source_embeddings.shape[1])),
                ),
                index=self.source_embeddings.index,
            )
        self.X = self.source_embeddings.loc[self.labels["source"]].values
        self.y = self.target_embeddings.loc[self.labels["target"]].values

    def train(self):
        u, _, vt = np.linalg.svd(self.y.T.dot(self.X))
        W = u.dot(vt)
        state_dict = self.model.model.state_dict()
        state_dict["weight"] = torch.from_numpy(W).float()
        self.model.model.load_state_dict(state_dict)

    def save_model(self, path):
        torch.save(self.model, path)

    def load_model(self, path):
        self.model = torch.load(path)
