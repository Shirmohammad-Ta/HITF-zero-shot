

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Selector:
    def __init__(self, embeddings, threshold=0.75):
        self.embeddings = embeddings  # dict: {example_id: embedding}
        self.threshold = threshold

    def compute_lambda(self, task_embedding):
        similarities = cosine_similarity([task_embedding], list(self.embeddings.values()))[0]
        avg_sim = np.mean(similarities)
        return min(1.0, max(0.0, avg_sim))  # λ ∈ [0, 1]

    def select_examples(self, task_embedding, k=8):
        sims = cosine_similarity([task_embedding], list(self.embeddings.values()))[0]
        sorted_indices = np.argsort(sims)[::-1][:k]
        return sorted_indices
