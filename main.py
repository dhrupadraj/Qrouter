import pandas as pd
from agent.q_agent import QAgent
from ticket_env.ticket_env import TicketEnvironment
from utils.feature_extraction import extract_features
from utils.metrics import print_metrics

# Actions = Teams
teams = ["Support", "Billing", "Product"]

# Load dataset
data = pd.read_csv("data/emails.csv")

# Environment
env = TicketEnvironment(data, extract_features, teams)

# State size = number of discrete states (0–4 → 5 states)
state_size = 5
action_size = len(teams)

# Q-Learning agent
agent = QAgent(state_size, action_size)

episodes = 50

for ep in range(episodes):
    state = env.reset()          # state is now an integer, not a vector
    total_rewards = []

    while state is not None:
        action = agent.choose_action(state)

        next_state, reward, done = env.step(action)

        # If next_state is None, environment ended → treat as terminal
        next_state = next_state if next_state is not None else 0

        agent.update(state, action, reward, next_state)

        total_rewards.append(reward)

        if done:
            break

        state = next_state

    print(f"Episode {ep+1}/{episodes}")
    print_metrics(total_rewards)

# Save Q-table
agent.save()
print("Training complete. Q-table saved.")
