from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from main import load_live_agent, predict_live_email

app = FastAPI()

# Load agent once at startup
agent, teams = load_live_agent()


class Email(BaseModel):
    subject: str
    body: str


@app.post("/predict")
def predict(email: Email):

    predicted_team = predict_live_email(
        subject=email.subject,
        body=email.body
    )

    return {"team": predicted_team}
@app.get("/")
def home():
    return {"message": "QRouter API is running"}
