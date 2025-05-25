from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('wine_quality_model.joblib')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Define feature names in correct order
FEATURE_NAMES = [
    'fixed acidity', 'volatile acidity', 'citric acid', 
    'residual sugar', 'chlorides', 'free sulfur dioxide',
    'total sulfur dioxide', 'density', 'pH', 'sulfates',
    'alcohol', 'type'
]

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions"""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Create DataFrame with correct feature order
        input_data = pd.DataFrame([[
            data.get('fixed acidity', 0),
            data.get('volatile acidity', 0),
            data.get('citric acid', 0),
            data.get('residual sugar', 0),
            data.get('chlorides', 0),
            data.get('free sulfur dioxide', 0),
            data.get('total sulfur dioxide', 0),
            data.get('density', 0),
            data.get('pH', 0),
            data.get('sulfates', 0),
            data.get('alcohol', 0),
            0 if data.get('type', '').lower() == 'red' else 1
        ]], columns=FEATURE_NAMES)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Return prediction
        return jsonify({
            'prediction': float(prediction[0]),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    """Homepage with instructions"""
    return """
    <h1>Wine Quality Prediction API</h1>
    <p>Send a POST request to /predict with JSON data containing wine features:</p>
    <pre>
    {
        "fixed acidity": 7.4,
        "volatile acidity": 0.7,
        "citric acid": 0,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11,
        "total sulfur dioxide": 34,
        "density": 0.9978,
        "pH": 3.51,
        "sulfates": 0.56,
        "alcohol": 9.4,
        "type": "red"
    }
    </pre>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
