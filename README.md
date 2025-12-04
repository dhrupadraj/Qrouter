📧 Q-Router: AI-Powered Email Routing using Reinforcement Learning

Automatically classify and route incoming Gmail messages to teams using a DQN Reinforcement Learning model, FastAPI backend, and n8n automation.

Use this link to try out the n8n workflow: http://localhost:5678/workflow/i6lmTGjOVchEVYjm

🚀 Overview

Q-Router is an intelligent email routing system that learns how to classify and assign incoming emails to the correct team (e.g., Billing, Product, Support, Technical) using:

Deep Q-Learning (DQN) for classification

Semantic / TF-IDF embeddings for text representation

FastAPI for real-time inference

n8n for workflow automation

Gmail Trigger to fetch live email

Slack Messaging to notify relevant teams

The system is trained on a dataset (emails.csv) containing email subjects, bodies, and correct team labels.
Once trained, the model is deployed to handle live Gmail traffic and send real-time Slack alerts.

🧠 Core Features
✔️ Reinforcement Learning-based Email Routing

Uses a Deep Q-Network (DQN) to learn optimal routing decisions from historical email assignments.

✔️ Real-Time Gmail Integration

Incoming emails are captured using n8n’s Gmail Trigger and forwarded to the FastAPI model.

✔️ Slack Team Notifications

Based on the predicted team (e.g., Product, Billing…), Slack messages are automatically delivered.

✔️ Embedding-Based Email Understanding

Supports:

Sentence Transformers (BERT-style embeddings)

TF-IDF fallback

✔️ Fully Deployment-Ready

Includes:

model.pt (trained PyTorch model)

teams.pkl (mapping of RL actions → team names)

FastAPI serve.py

n8n workflow config
