
# hyperparameters and paths
SEED = 42

DATA_PATH = "data/emails.csv"
MODEL_PATH = "models/dqn.pth"
RESULTS_CSV = "results/routing_predictions.csv"

# DQN params
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
REPLAY_CAPACITY = 10000
TARGET_UPDATE = 500   # steps
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
NUM_EPISODES = 50
MAX_STEPS_PER_EPISODE = 50

# Feature / embedding
USE_SENTENCE_TRANSFORMERS = True   # set False to force TF-IDF fallback
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Device
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

