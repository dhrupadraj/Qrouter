from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from config import USE_SENTENCE_TRANSFORMERS   
from main import load_live_agent
from utils.feature_extraction import extract_features, _load_sentence_model

app = FastAPI(title="QRouter API", version="1.0.0")

# Enable CORS for n8n and other clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables - initialized in startup event
agent = None
teams = None
startup_error = None

@app.on_event("startup")
async def startup_event():
    """Pre-load everything at startup to avoid delays on first request"""
    global agent, teams, startup_error
    try:
        print("🔄 Loading agent...")
        agent, teams = load_live_agent()
        print(f"✅ Agent loaded successfully with {len(teams)} teams")
        
        # Pre-load sentence transformer model if using it
        if USE_SENTENCE_TRANSFORMERS:
            print("🔄 Pre-loading sentence transformer model...")
            _load_sentence_model()
            # Test it with a dummy input to ensure it works
            test_emb = extract_features("test", fit_tfidf_if_needed=False)
            print(f"✅ Sentence transformer model loaded (embedding dim: {test_emb.shape[0]})")
        
        print("✅ Server ready to accept requests!")
        startup_error = None
    except Exception as e:
        startup_error = str(e)
        print(f"❌ Failed to load during startup: {e}")
        import traceback
        traceback.print_exc()
        # Don't raise - let server start so we can return error messages


class Email(BaseModel):
    subject: str
    body: str


@app.post("/predict")
def predict(email: Email):
    """
    Predict the team for an email based on subject and body.
    Expected JSON: {"subject": "...", "body": "..."}
    """
    try:
        # Validate input
        if not email.subject and not email.body:
            raise HTTPException(status_code=400, detail="Both subject and body cannot be empty")
        
        # Use the pre-loaded agent instead of reloading
        text = (email.subject or "") + " " + (email.body or "")
        emb = extract_features(text, fit_tfidf_if_needed=False)
        
        action = agent.select_action(emb, greedy=True)
        predicted_team = teams[action]
        
        return {"team": predicted_team}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/")
def home():
    return {"message": "QRouter API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "teams_count": len(teams)}
