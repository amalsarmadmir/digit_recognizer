# MNIST Digit Predictor

A Flask web application that uses a pre-trained Keras model to predict handwritten digits from uploaded images.

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Setup

1. Clone or download this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Place your trained MNIST model file (model.h5) in the root directory of the project

## Running the Application

1. Make sure your virtual environment is activated
2. Run the Flask application:
   ```bash
   python app.py
   ```
3. Open your web browser and navigate to `http://localhost:5000`

## Usage

1. Click the "Choose Image" button to select an image file
2. The application will display a preview of your image
3. The predicted digit and confidence score will be shown below the image

## Notes

- The application expects images to be in common formats (PNG, JPG, JPEG)
- The model expects input similar to MNIST dataset (grayscale images)
- The application will automatically preprocess your image to match the MNIST format (28x28 pixels, grayscale)

## Troubleshooting

- If you see "Model not loaded" error, ensure that the model.h5 file is present in the root directory
- Make sure all dependencies are installed correctly
- Check the console for any error messages 