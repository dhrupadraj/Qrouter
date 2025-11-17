import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

def print_episode_stats(ep, reward, loss):
    print(f"Episode {ep:4d} | Total reward: {reward:6.2f} | Loss: {loss:.4f}")

def plot_rewards(rewards, path=None):
    plt.figure(figsize=(10,4))
    plt.plot(rewards, label="episode reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    if path:
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.show()

def compute_confusion(true_labels, pred_labels, labels):
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    acc = accuracy_score(true_labels, pred_labels)
    return cm, acc
