from datasets import Dataset
from dgeb import modality
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from functools import partial
import matplotlib.pyplot as plt
from dgeb.evaluators import ClusteringEvaluator
from datasets import load_dataset



class FourierEmbeddingModel():
    def __init__(self, F=64, H=128):
        self.F = F
        self.H = H

   

    @property
    def embed_dim(self):
        return self.F  

    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def eq_2(self, x, W):
        D = x.shape[1]
        return (1 / np.sqrt(D)) * np.concatenate([np.cos(x @ W.T), np.sin(x @ W.T)], axis=1)

    def two_layer_NN(self, F, W1, W2, B1, B2):
        return self.gelu(F @ W1 + B1) @ W2 + B2

    def get_tokenizations(self, sequences: list[str], max_length=None) -> np.ndarray:
        sequence_to_vector = {'A': [1,0,0,0], 'C': [0,1,0,0], 'T': [0,0,1,0], 'G': [0,0,0,1]}

        if max_length is None:
            max_length = max(len(seq) for seq in sequences)

        resulting_tokenization = []
        for sequence in sequences:
            tokenization = [sequence_to_vector[elem] for elem in sequence]

            while len(tokenization) < max_length:
                tokenization.append([0,0,0,0])

            tokenization = tokenization[:max_length]

            resulting_tokenization.append(tokenization)

        return torch.tensor(resulting_tokenization, dtype=torch.float32)  
    
    def encode(self, sequences, **kwargs):
        x = self.get_tokenizations(sequences)
        N, G, M = x.shape

        W = np.random.randn(self.F // 2, M)

        W1 = np.random.randn(self.F, self.H)
        B1 = np.random.randn(self.H)
        D_out = G * 16
        W2 = np.random.randn(self.H, D_out // G)
        B2 = np.random.randn(D_out // G)

        x_flat = x.reshape(-1, M)
        F_out = self.eq_2(x_flat, W)
        Y = self.two_layer_NN(F_out, W1, W2, B1, B2)
        embeddings = np.reshape(Y, (N, D_out))

        # Return in [N, num_layers, embed_dim] format
        return embeddings[:, None, :]  # shape [N, 1, D_out]


from dgeb.evaluators import ClusteringEvaluator
from datasets import load_dataset

def compute_v_scores():
    dataset = load_dataset("tattabio/e_coli_rnas")["train"]
    sequences = dataset["Sequence"]
    labels = dataset["Label"]
    fourier_embedding_model_v__scores = []
    for i in range(50):
        fourier_model = FourierEmbeddingModel()
        embeddings = fourier_model.encode(sequences)

        evaluator = ClusteringEvaluator(embeds=embeddings[:, 0], labels=labels)
        fourier_embedding_model_v__scores.append(evaluator())
    return fourier_embedding_model_v__scores

def plot_v_scores():
    fourier_embedding_v_scores = compute_v_scores()
    v_scores_list = [dict['v_measure'] for dict in fourier_embedding_v_scores]
    plt.figure(figsize=(8, 6))
    plt.plot(v_scores_list)

    # Add labels and title
    plt.xlabel('Trial Number', fontsize=12)
    plt.ylabel('V-score', fontsize=12)
    plt.title('V-scores for 50 trials', fontsize=14)
    plt.grid(True)

    # Show the plot
    plt.show()

plot_v_scores()