from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import gdown
import os

# Initialize Flask app
app = Flask(__name__)

# Load the model
model_path = "plant_disease_model.keras"

# If model doesn't exist locally, download it
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1OyByJOLdcEKofoUs6B1-AdWsEkPsT6l-" 
    gdown.download(url, model_path, quiet=False)

# Load the model
model = load_model(model_path)

# Class names grouped by plant types (improved names)
CLASS_NAMES = {
    'Apple': {
        0: 'Apple scab',
        1: 'Black rot',
        2: 'Cedar apple rust',
        3: 'Healthy'
    },
    'Blueberry': {
        4: 'Healthy'
    },
    'Cherry': {
        5: 'Powdery mildew',
        6: 'Healthy'
    },
    'Corn': {
        7: 'Cercospora leaf spot Gray leaf spot',
        8: 'Common rust',
        9: 'Northern Leaf Blight',
        10: 'Healthy'
    },
    'Grape': {
        11: 'Black rot',
        12: 'Esca (Black Measles)',
        13: 'Leaf blight (Isariopsis Leaf Spot)',
        14: 'Healthy'
    },
    'Orange': {
        15: 'Haunglongbing (Citrus greening)'
    },
    'Peach': {
        16: 'Bacterial spot',
        17: 'Healthy'
    },
    'Pepper': {
        18: 'Bacterial spot',
        19: 'Healthy'
    },
    'Potato': {
        20: 'Early blight',
        21: 'Late blight',
        22: 'Healthy'
    },
    'Raspberry': {
        23: 'Healthy'
    },
    'Soybean': {
        24: 'Healthy'
    },
    'Squash': {
        25: 'Powdery mildew'
    },
    'Strawberry': {
        26: 'Leaf scorch',
        27: 'Healthy'
    },
    'Tomato': {
        28: 'Bacterial spot',
        29: 'Early blight',
        30: 'Late blight',
        31: 'Leaf Mold',
        32: 'Septoria leaf spot',
        33: 'Spider mites Two-spotted spider mite',
        34: 'Target Spot',
        35: 'Tomato Yellow Leaf Curl Virus',
        36: 'Tomato mosaic virus',
        37: 'Healthy'
    }
}

# Preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to 224x224
    image = np.array(image).astype('float32') / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Routes
@app.route('/')
def index():
    return render_template('index.html', class_names=CLASS_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the file exists in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})

        file = request.files['file']

        # Open the image and preprocess
        image = Image.open(file)
        processed_image = preprocess_image(image)

        # Perform prediction
        prediction = model.predict(processed_image)
        class_idx = np.argmax(prediction)
        confidence = prediction[0][class_idx]

        # Find the disease name based on predicted index
        plant_type = None
        predicted_label = None
        for category, classes in CLASS_NAMES.items():
            if class_idx in classes:
                plant_type = category
                predicted_label = classes[class_idx]
                break

        return jsonify({
            'plant_type': plant_type,
            'disease': predicted_label,
            'confidence': float(confidence)
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)