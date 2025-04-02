from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS  # Import CORS

# Load the trained model
model = joblib.load('spam_classifier_model.pkl')

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

@app.route('/')
def home():
    return "Spam Email Classifier API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    email_text = [data['text']]
    prediction = model.predict(email_text)
    result = "spam" if prediction[0] == 1 else "not-spam"
    
    return jsonify({'result': result})  # Changed response key to match frontend

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
