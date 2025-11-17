import numpy as np
import pandas as pd

class TicketEnvironment:
    """
    Environment that yields embeddings (states) for each email, accepts actions (team indices),
    and returns reward and next_state. This environment is episodic: one episode processes
    the entire dataset (optionally shuffled each episode).
    """

    def __init__(self, df: pd.DataFrame, feature_extractor, teams, shuffle=True):
        """
        df: DataFrame with columns: 'subject', 'body', 'true_team'
        feature_extractor: callable(text) -> 1D numpy vector
        teams: list of team names in order (action indices)
        """
        self.original_df = df.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        self.teams = list(teams)
        self.shuffle = shuffle
        self.reset()

    def reset(self):
        # shuffle the data each episode so agent can't memorize ordering
        if self.shuffle:
            self.df = self.original_df.sample(frac=1).reset_index(drop=True)
        else:
            self.df = self.original_df.copy()
        self.i = 0
        # precompute features to speed up training
        corpus = (self.df["subject"].fillna("") + " " + self.df["body"].fillna("")).tolist()
        # If TF-IDF fallback is needed outside extractor, you should have fitted vectorizer externally.
        self.features = [self.feature_extractor(text) for text in corpus]
        return self._get_state()

    def _get_state(self):
        if self.i >= len(self.df):
            return None
        return self.features[self.i]

    def step(self, action):
        """
        action: integer index of chosen team
        returns: (next_state, reward, done, info)
        """
        if self.i >= len(self.df):
            return None, 0.0, True, {}

        true_team = self.df.loc[self.i, "true_team"]
        predicted_team = self.teams[action] if 0 <= action < len(self.teams) else None

        # reward shaping: strong positive for correct, mild negative for wrong
        reward = 5.0 if predicted_team == true_team else -1.0

        self.i += 1
        done = self.i >= len(self.df)
        next_state = self._get_state() if not done else None
        return next_state, reward, done, {"true_team": true_team, "predicted_team": predicted_team}
