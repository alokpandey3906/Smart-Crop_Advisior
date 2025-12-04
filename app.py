from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

from openai import OpenAI
import google.generativeai as genai

app = Flask(__name__)
CORS(app)  # allow your HTML page to call this API


# ========== API KEYS ==========
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # set in your system env
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # set in your system env

# TEMP (only for local testing – do NOT commit real keys to GitHub)
# OPENAI_API_KEY = "YOUR_OPENAI_KEY_HERE"
# GEMINI_API_KEY = "YOUR_GEMINI_KEY_HERE"

openai_client = OpenAI(api_key=OPENAI_API_KEY)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
# ==============================


# ========== LOAD ML MODEL (for /predict) ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "crop_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print("Error loading crop_model.pkl:", e)
    model = None
# ==================================================


@app.route("/")
def home():
    return "Smart Crop Advisor API is running."


# ========== PREDICTION ENDPOINT ==========
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "ML model not loaded on server"}), 500

    data = request.get_json() or {}

    try:
        N = float(data["N"])
        P = float(data["P"])
        K = float(data["K"])
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        ph = float(data["ph"])
        rainfall = float(data["rainfall"])
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid or missing input fields: {e}"}), 400

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    predicted_crop = model.predict(features)[0]

    return jsonify({"recommended_crop": predicted_crop})
# =========================================


# ========== HELPERS: ASK OPENAI / GEMINI ==========
def ask_openai(message: str):
    if not OPENAI_API_KEY:
        return None, "OPENAI_API_KEY is not set on the server."

    messages = [
        {
            "role": "system",
            "content": (
                "You are an agricultural expert AI helping Indian farmers. "
                "Give clear, practical advice in simple language (you can mix Hindi + English). "
                "Focus on crops, soil, fertilizers, irrigation, pests, and practical steps."
            ),
        },
        {"role": "user", "content": message},
    ]

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=messages,
            temperature=0.7,
        )
        answer = completion.choices[0].message.content
        return answer, None
    except Exception as e:
        print("OpenAI error:", e)
        return None, str(e)


def ask_gemini(message: str):
    if not GEMINI_API_KEY:
        return None, "GEMINI_API_KEY is not set on the server."

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # or "gemini-1.5-pro"
        resp = model.generate_content(message)
        # resp.text should contain the main answer
        return resp.text, None
    except Exception as e:
        print("Gemini error:", e)
        return None, str(e)
# ===================================================


# ========== CHAT ENDPOINTS ==========

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Expects JSON:
      {
        "message": "user question",
        "provider": "openai" | "gemini"   (optional, default = openai)
      }
    """
    data = request.get_json() or {}

    user_message = (data.get("message") or data.get("question") or "").strip()
    provider = (data.get("provider") or "openai").lower()

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    if provider == "gemini":
        answer, error = ask_gemini(user_message)
        provider_used = "gemini"
    else:
        answer, error = ask_openai(user_message)
        provider_used = "openai"

    if error:
        return jsonify(
            {
                "answer": "Sorry, I could not get a reply from the AI service.",
                "error": error,
                "provider": provider_used,
            }
        ), 500

    return jsonify({"answer": answer, "provider": provider_used})


# Optional older route, always uses OpenAI
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_message = (data.get("question") or data.get("message") or "").strip()

    if not user_message:
        return jsonify({"answer": "Please ask a valid farming question."}), 400

    answer, error = ask_openai(user_message)

    if error:
        return jsonify(
            {
                "answer": "Sorry, I could not reach the AI service.",
                "error": error,
            }
        ), 500

    return jsonify({"answer": answer, "provider": "openai"})


# ====================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
