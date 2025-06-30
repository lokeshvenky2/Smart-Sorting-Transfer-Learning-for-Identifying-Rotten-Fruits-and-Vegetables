import os
import json
import logging
import traceback
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import io

from config import Config

# ----------------------------------
# Configure Logging
# ----------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------------
# Initialize Flask App
# ----------------------------------
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# ----------------------------------
# Global Variables
# ----------------------------------
model = None
index_to_class = None

# ----------------------------------
# Load Model and Class Indices
# ----------------------------------
def load_ml_assets():
    global model, index_to_class
    try:
        if not os.path.exists(Config.MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {Config.MODEL_PATH}")
        if not os.path.exists(Config.CLASS_INDICES_PATH):
            raise FileNotFoundError(f"Class indices file not found at: {Config.CLASS_INDICES_PATH}")

        model = load_model(Config.MODEL_PATH)
        with open(Config.CLASS_INDICES_PATH, 'r') as f:
            class_indices = json.load(f)

        # Convert string/int keys safely
        index_to_class = {int(v): k for k, v in class_indices.items()}
        logger.info("✅ ML model and class indices loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Error loading ML assets: {e}", exc_info=True)
        model = None
        index_to_class = None

# ----------------------------------
# Startup Model Load
# ----------------------------------
with app.app_context():
    load_ml_assets()

# ----------------------------------
# Routes
# ----------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or index_to_class is None:
        return jsonify({"error": "Server not ready: model not loaded"}), 503

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Load and preprocess image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((Config.IMG_WIDTH, Config.IMG_HEIGHT))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        preds = model.predict(img_array)
        predicted_index = int(np.argmax(preds[0]))
        predicted_class = index_to_class.get(predicted_index, "Unknown")
        confidence = float(preds[0][predicted_index]) * 100.0

        logger.info(f"🔍 Predicted: {predicted_class} ({confidence:.2f}%)")

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "all_probabilities": [round(float(p) * 100.0, 2) for p in preds[0]]
        })

    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"🔥 Prediction failed:\n{traceback_str}")
        return jsonify({"error": f"Internal error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    status = "healthy" if model is not None and index_to_class is not None else "unhealthy"
    message = "ML model and assets loaded." if status == "healthy" else "ML model or assets failed to load."
    return jsonify({"status": status, "message": message}), 200 if status == "healthy" else 500

# ----------------------------------
# Entry Point
# ----------------------------------
if __name__ == '__main__':
    app.run(debug=True)
