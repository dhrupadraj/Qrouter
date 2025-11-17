import os
import argparse
import numpy as np
import pandas as pd
from tqdm import trange

from config import *
from utils.feature_extraction import extract_features, fit_tfidf
from ticket_env.ticket_env import TicketEnvironment
from agent.dqn_agent import DQNAgent
from utils.metrics import print_episode_stats, plot_rewards, compute_confusion

# load data
def load_data(path):
    df = pd.read_csv(path)

    if 'subject' not in df.columns or 'body' not in df.columns or 'true_team' not in df.columns:
        raise ValueError("CSV must contain 'subject', 'body', 'true_team' columns.")

    df['subject'] = df['subject'].fillna('')
    df['body'] = df['body'].fillna('')

    # FIX: normalize labels
    df['true_team'] = df['true_team'].astype(str).str.strip()
    df['true_team'] = df['true_team'].replace({'nan': 'Unknown', 'None': 'Unknown'})

    return df


def prepare_feature_extractor(df):
    # If TF-IDF fallback is needed, fit the vectorizer
    if not USE_SENTENCE_TRANSFORMERS:
        corpus = (df['subject'] + ' ' + df['body']).tolist()
        fit_tfidf(corpus)
    return

def train(args):
    df = load_data(DATA_PATH)
    teams = sorted(df['true_team'].unique().tolist())
    env = TicketEnvironment(df, lambda t: extract_features(t, fit_tfidf_if_needed=False), teams, shuffle=True)

    # determine input dimension by extracting first embedding
    sample_text = df.iloc[0]['subject'] + " " + df.iloc[0]['body']
    sample_vec = extract_features(sample_text, fit_tfidf_if_needed=False)
    input_dim = sample_vec.shape[0]
    n_actions = len(teams)

    agent = DQNAgent(input_dim, n_actions)
    rewards_log = []
    losses_log = []

    global_step = 0
    for ep in trange(NUM_EPISODES, desc="Episodes"):
        state = env.reset()
        ep_reward = 0.0
        ep_losses = []

        step = 0
        while True:
            action = agent.select_action(state, greedy=False)
            next_state, reward, done, info = env.step(action)

            agent.store(state, action, reward, next_state if next_state is not None else None, done)
            loss = agent.update()
            if loss:
                ep_losses.append(loss)

            ep_reward += reward
            global_step += 1
            step += 1
            if done or step >= MAX_STEPS_PER_EPISODE:
                break
            state = next_state

        rewards_log.append(ep_reward)
        losses_log.append(np.mean(ep_losses) if ep_losses else 0.0)

        if (ep + 1) % 10 == 0:
            print_episode_stats(ep + 1, ep_reward, losses_log[-1])
        # checkpoint periodically
        if (ep + 1) % 50 == 0:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            agent.save(MODEL_PATH)

    # final save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    agent.save(MODEL_PATH)
    plot_rewards(rewards_log)
    return agent, env, teams

def evaluate(agent, env, teams, save_csv=True):
    # run greedy pass over the original dataset
    df = env.original_df
    preds = []
    trues = []
    for i in range(len(df)):
        text = df.loc[i, 'subject'] + " " + df.loc[i, 'body']
        state = extract_features(text, fit_tfidf_if_needed=False)
        a = agent.select_action(state, greedy=True)
        preds.append(teams[a])
        trues.append(df.loc[i, 'true_team'])
    df_out = df.copy()
    df_out['predicted_team'] = preds
    if save_csv:
        os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
        df_out.to_csv(RESULTS_CSV, index=False)
        print(f"Saved predictions to {RESULTS_CSV}")

    cm, acc = compute_confusion(trues, preds, labels=teams)
    print("Accuracy:", acc)
    print("Confusion matrix:")
    print(cm)
    return df_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train DQN")
    parser.add_argument("--eval", action="store_true", help="Evaluate using saved model")
    args = parser.parse_args()

    # prepare TF-IDF if needed
    df_all = load_data(DATA_PATH)
    if not USE_SENTENCE_TRANSFORMERS:
        prepare_feature_extractor(df_all)

    if args.train:
        agent, env, teams = train(args)
    elif args.eval:
        # load model and evaluate
        teams = sorted(df_all['true_team'].unique().tolist())
        sample_text = df_all.iloc[0]['subject'] + " " + df_all.iloc[0]['body']
        sample_vec = extract_features(sample_text, fit_tfidf_if_needed=False)
        input_dim = sample_vec.shape[0]
        n_actions = len(teams)
        agent = DQNAgent(input_dim, n_actions)
        agent.load(MODEL_PATH)
        env = TicketEnvironment(df_all, lambda t: extract_features(t, fit_tfidf_if_needed=False), teams, shuffle=False)
        evaluate(agent, env, teams, save_csv=True)
    else:
        print("Run with --train or --eval")
