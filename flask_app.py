import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import time

# --- Setup ---
app = Flask(__name__)
CORS(app)

# --- Load AI/ML Models and Preprocessors ---
# This code runs only once when the server starts.
print("Loading machine learning models...")
try:
    model = joblib.load('/home/Begad/mysite/diagnostic_model.pkl')
    scaler = joblib.load('/home/Begad/mysite/scaler.pkl')
    label_encoder = joblib.load('/home/Begad/mysite/label_encoder.pkl')
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    model, scaler, label_encoder = None, None, None

# --- OpenAI Client Setup ---
client = OpenAI(api_key="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") # Using the hardcoded key for now

# --- Home Route ---
@app.route("/")
def home():
    return "AI Diagnoser API is running!"

# --- HPI Questions Endpoint (OpenAI LLM) ---
@app.route("/generate-hpi-questions/", methods=['POST'])
def generate_hpi_questions():
    # ... (This function remains the same)
    pass # Placeholder for your existing code

# --- NEW: Final Diagnosis Endpoint (Your XGBoost Model) ---
@app.route("/predict-diagnosis/", methods=['POST'])
def predict_diagnosis():
    if not all([model, scaler, label_encoder]):
        return jsonify({"error": "Models are not loaded. Check server logs."}), 500

    try:
        # 1. Get the feature data from the app's request
        input_data = request.get_json()

        # Convert the incoming JSON data into a pandas DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure the order of columns in the DataFrame matches the order used during training
        # You'll need to get this list from your training script.
        # training_columns = [...] # The list of column names from your X_train
        # input_df = input_df[training_columns]

        # 2. Scale the features using the loaded scaler
        input_scaled = scaler.transform(input_df)

        # 3. Make a prediction
        prediction_encoded = model.predict(input_scaled)

        # 4. Decode the prediction back to the original ICD code
        prediction_label = label_encoder.inverse_transform(prediction_encoded)

        # 5. Return the result
        return jsonify({"predicted_diagnosis": prediction_label[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 400
