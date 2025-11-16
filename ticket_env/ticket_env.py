import numpy as np

class TicketEnvironment:
    def __init__(self, data, feature_extractor, teams):
        self.data = data
        self.feature_extractor = feature_extractor
        self.teams = teams
        self.current_idx = 0

    def reset(self):
        self.current_idx = 0
        return self._get_state()

    def _get_state(self):
        if self.current_idx >= len(self.data):
            return None
        email = self.data.iloc[self.current_idx]
        text = f"{email['subject']} {email['body']}"
        return self.feature_extractor(text)

    def step(self, action):
        email = self.data.iloc[self.current_idx]
        correct_team = email['true_team']
        reward = 10 if self.teams[action] == correct_team else -1

        self.current_idx += 1
        done = self.current_idx >= len(self.data)
        next_state = self._get_state() if not done else None

        return next_state, reward, done
