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
    def __init__(self, F=8, H=16, D=16):
        self.F = F
        self.H = H
        self.D = D 

   


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

        W_r = np.random.randn(self.F // 2, M)

        W1 = np.random.randn(self.F, self.H)
        B1 = np.random.randn(self.H)
        D = G * self.D
        W2 = np.random.randn(self.H, D // G)
        B2 = np.random.randn(D // G)

        x_flat = x.reshape(-1, M)
        F_out = self.eq_2(x_flat, W_r)
        Y = self.two_layer_NN(F_out, W1, W2, B1, B2)
        embeddings = np.reshape(Y, (N, D))

        return np.expand_dims(embeddings, axis=1)




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
    print(fourier_embedding_v_scores)
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


def hyperparameter_grid_search():
    values = [8, 16, 32, 64, 128]
    dataset = load_dataset("tattabio/e_coli_rnas")["train"]
    sequences = dataset["Sequence"]
    labels = dataset["Label"]
    max_indices = ()
    max = 0
    v_scores_list = []
    for D in values:
        for H in values:
            for F in values:
                fourier_model = FourierEmbeddingModel(F, H, D)
                embeddings = fourier_model.encode(sequences)

                evaluator = ClusteringEvaluator(embeds=embeddings[:, 0], labels=labels)
                v_measure = evaluator()
                print(v_measure)
                if v_measure['v_measure'] > max:
                    max = v_measure['v_measure']
                    max_indices = (D,H,F)
                v_scores_list.append(v_measure['v_measure'])
    print(f'v-measure score for the values of hyperparameters is {max_indices} with {max}')
    print(f'statistics of hyperparameters are maximum: {max} with mean: {np.mean(v_scores_list)} median: {np.median(v_scores_list)}')
    plt.figure(figsize=(8, 6))
    plt.boxplot(v_scores_list)


    # Add labels and title
    plt.xlabel('Trial Number', fontsize=12)
    plt.ylabel('V-score', fontsize=12)
    plt.title('V-scores for each trial', fontsize=14)
    plt.grid(True)

    # Show the plot
    plt.show()



hyperparameter_grid_search()