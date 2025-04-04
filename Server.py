from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import sys

# Optional Windows-specific logic
if sys.platform == "win32":
    try:
        import win32api
    except ImportError:
        print("pywin32 not installed, but running on Windows.")
else:
    print("Running on a non-Windows environment.")

# Load model safely with full path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'spam_classifier_model.pkl')
model = joblib.load(MODEL_PATH)

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

@app.route('/')
def home():
    return jsonify({"message": "Spam Email Classifier API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": "Invalid JSON"}), 400

    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    email_text = [data['text']]
    try:
        prediction = model.predict(email_text)
        result = "spam" if prediction[0] == 1 else "not-spam"
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
