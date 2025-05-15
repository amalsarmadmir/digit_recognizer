from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import logging
import os

port = int(os.environ.get("PORT", 8080))  # Railway sets this PORT automatically
app.run(host="0.0.0.0", port=port)


app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the pre-trained model
try:
    model = load_model('model.h5')
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    model = None

def preprocess_image(image_data):
    """Preprocess the uploaded image for prediction."""
    try:
        # Convert the image to grayscale and resize to 28x28
        img = Image.open(io.BytesIO(image_data)).convert('L')
        img = img.resize((28, 28), Image.Resampling.LANCZOS)  # Best resampling in PIL 10+
        
        # Normalize the image
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    except Exception as e:
        logging.error(f"Image preprocessing failed: {e}")
        raise ValueError("Invalid image format or corrupted image.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        image_data = file.read()
        processed_image = preprocess_image(image_data)

        prediction = model.predict(processed_image, verbose=0)
        predicted_digit = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][predicted_digit])
        
        return jsonify({
            'prediction': predicted_digit,
            'confidence': round(confidence * 100, 2)
        })
    except Exception as e:
        logging.exception("Prediction error:")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
