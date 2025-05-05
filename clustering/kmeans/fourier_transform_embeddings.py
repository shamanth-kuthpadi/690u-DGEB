from datasets import Dataset
from dgeb import modality
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from functools import partial


class FourierEmbeddingModel():
    def __init__(self, F=64, H=128, **kwargs):
        self.F = F
        self.H = H

    @property
    def modality(self):
        return modality.DNA

    @property
    def num_layers(self):
        return 1  # Single layer of learned features

    @property
    def embed_dim(self):
        return self.F  # Or whatever D_out you use

    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def eq_2(self, x, W):
        D = x.shape[-1]
        return (1 / np.sqrt(D)) * np.concatenate([np.cos(x @ W.T), np.sin(x @ W.T)], axis=-1)

    def two_layer_NN(self, F, W1, W2, B1, B2):
        return self.gelu(F @ W1 + B1) @ W2 + B2

    def get_tokenizations(self, sequences: list[str], max_length=None) -> np.ndarray:
        sequence_to_vector = {'A': [1,0,0,0], 'C': [0,1,0,0], 'T': [0,0,1,0], 'G': [0,0,0,1], 'PAD': [0,0,0,0]}

        if max_length is None:
            max_length = max(len(seq) for seq in sequences)

        resulting_tokenization = []
        for sequence in sequences:
            tokenization = [sequence_to_vector[elem] for elem in sequence]

            # Pad with 'PAD' vectors if sequence is shorter than max_length
            while len(tokenization) < max_length:
                tokenization.append(sequence_to_vector['PAD'])

            # Truncate if longer
            tokenization = tokenization[:max_length]

            resulting_tokenization.append(tokenization)

        return torch.tensor(resulting_tokenization, dtype=torch.float32)  # shape: [N, G, M]

    def encode(self, sequences, **kwargs):
        x = self.get_tokenizations(sequences)  # [N, G, M]
        N, G, M = x.shape

        # Random Fourier transform weights
        W = np.random.randn(self.F // 2, M)

        # Random two-layer NN weights
        W1 = np.random.randn(self.F, self.H)
        B1 = np.random.randn(self.H)
        D_out = G * 16
        W2 = np.random.randn(self.H, D_out // G)
        B2 = np.random.randn(D_out // G)

        # Apply eq_2 + 2-layer MLP
        x_flat = x.reshape(-1, M)
        F_out = self.eq_2(x_flat, W)
        Y = self.two_layer_NN(F_out, W1, W2, B1, B2)
        embeddings = np.reshape(Y, (N, D_out))

        # Return in [N, num_layers, embed_dim] format
        return embeddings[:, None, :]  # shape [N, 1, D_out]


from dgeb.evaluators import ClusteringEvaluator
from datasets import load_dataset

dataset = load_dataset("tattabio/e_coli_rnas")["train"]
sequences = dataset["Sequence"]
labels = dataset["Label"]

fourier_model = FourierEmbeddingModel()
embeddings = fourier_model.encode(sequences)

evaluator = ClusteringEvaluator(embeds=embeddings[:, 0], labels=labels)
print(evaluator())
