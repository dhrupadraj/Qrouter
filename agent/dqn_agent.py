import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import SEED, GAMMA, LR, BATCH_SIZE, REPLAY_CAPACITY, TARGET_UPDATE, DEVICE

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=REPLAY_CAPACITY):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(256,128)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, input_dim, n_actions, lr=LR, gamma=GAMMA, device=DEVICE):
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.gamma = gamma

        self.policy_net = QNetwork(input_dim, n_actions).to(self.device)
        self.target_net = QNetwork(input_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayBuffer()
        self.steps_done = 0

        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.995

    def select_action(self, state_vec, greedy=False):
        # state_vec: numpy array
        if not greedy and random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
            qvals = self.policy_net(s)
            return int(torch.argmax(qvals).item())

    def store(self, state, action, reward, next_state, done):
        # store numpy arrays directly
        self.replay.push(state, action, reward, next_state, done)

    def update(self, batch_size=BATCH_SIZE):
        if len(self.replay) < batch_size:
            return 0.0

        trans = self.replay.sample(batch_size)
        states = torch.from_numpy(np.vstack(trans.state)).float().to(self.device)
        actions = torch.tensor(trans.action, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(trans.reward, dtype=torch.float).unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(np.vstack([s if s is not None else np.zeros(states.shape[1], dtype=np.float32) for s in trans.next_state])).float().to(self.device)
        dones = torch.tensor(trans.done, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1)[0].unsqueeze(1)
            q_target = rewards + (1 - dones) * self.gamma * q_next

        loss = nn.MSELoss()(q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # epsilon decay
        self.eps = max(self.eps * self.eps_decay, self.eps_min)

        self.steps_done += 1
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def save(self, path):
        torch.save({
            'policy': self.policy_net.state_dict(),
            'target': self.target_net.state_dict(),
            'eps': self.eps
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt['policy'])
        self.target_net.load_state_dict(ckpt['target'])
        if 'eps' in ckpt:
            self.eps = ckpt['eps']
