import numpy as np
import pickle
import os

class QAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = np.zeros((state_size, action_size))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state_idx):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.q_table.shape[1])
        return np.argmax(self.q_table[state_idx])

    def update(self, s, a, r, s_next):
        best_next = np.max(self.q_table[s_next]) if s_next is not None else 0
        self.q_table[s, a] += self.alpha * (r + self.gamma * best_next - self.q_table[s, a])

    def save(self, path="models/q_table.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, path="models/q_table.pkl"):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)
