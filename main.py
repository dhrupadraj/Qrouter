import pandas as pd
import numpy as np
from agent.q_agent import QAgent
from ticket_env.ticket_env import TicketEnvironment
from utils.feature_extraction import extract_features
from utils.metrics import print_metrics

# Teams (actions)
teams = ["Support", "Billing", "Product","Technical"]

# Load email data
data = pd.read_csv("data/emails.csv")

# Environment setup
env = TicketEnvironment(data, extract_features, teams)

# Updated state size because we now use embeddings
# all-MiniLM-L6-v2 → vector size = 384
state_size = 384
action_size = len(teams)

# Q-learning agent
agent = QAgent(state_size, action_size)

# Training loop
episodes = 50
for ep in range(episodes):
    state = env.reset()
    total_rewards = []

    while state is not None:
        # For embeddings, using argmax is meaningless — use full vector
        s_vec = state

        # Convert vector to index via hashing (to map continuous → discrete)
        s_idx = hash(s_vec.tobytes()) % state_size

        action = agent.choose_action(s_idx)

        next_state, reward, done = env.step(action)

        if next_state is not None:
            s_next_vec = next_state
            s_next_idx = hash(s_next_vec.tobytes()) % state_size
        else:
            s_next_idx = 0

        agent.update(s_idx, action, reward, s_next_idx)
        total_rewards.append(reward)

        if done:
            break

        state = next_state

    print(f"Episode {ep+1}/{episodes}")
    print_metrics(total_rewards)

agent.save()

print("Training complete. Q-table saved.")
print("\n=== Email Routing Results ===\n")

for idx, row in data.iterrows():
    email_text = f"{row['subject']} {row['body']}"
    true_team = row["true_team"] if "true_team" in data.columns else (
                row["team"] if "team" in data.columns else "N/A")

    # Extract embedding
    state = extract_features(email_text)

    # Map embedding → Q-table index
    s_idx = hash(state.tobytes()) % state_size
    
    # Agent picks an action
    action = agent.choose_action(s_idx)
    predicted_team = teams[action]

    print(f"Email #{idx+1}")
    print(f"Subject: {row['subject'][:60]}")
    print(f"Predicted Team: {predicted_team}")
    print(f"Actual Team: {true_team}")
    print("-" * 50)
