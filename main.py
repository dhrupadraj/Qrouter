import pandas as pd
import numpy as np
from agent.q_agent import QAgent
from env.ticket_env import TicketEnvironment
from utils.feature_extraction import extract_features
from utils.metrics import print_metrics

# Teams (actions)
teams = ["Support", "Billing", "Product"]

# Load email data
data = pd.read_csv("data/emails.csv")

# Environment setup
env = TicketEnvironment(data, extract_features, teams)
state_size = 6  # number of keywords in feature extractor
action_size = len(teams)

# Q-learning agent
agent = QAgent(state_size, action_size)

# Training loop
episodes = 50
for ep in range(episodes):
    state = env.reset()
    total_rewards = []
    while state is not None:
        s_idx = np.argmax(state)
        action = agent.choose_action(s_idx)
        next_state, reward, done = env.step(action)
        s_next_idx = np.argmax(next_state) if next_state is not None else 0
        agent.update(s_idx, action, reward, s_next_idx)
        total_rewards.append(reward)
        if done:
            break
        state = next_state
    print(f"Episode {ep+1}/{episodes}")
    print_metrics(total_rewards)

agent.save()

print("Training complete. Q-table saved.")
