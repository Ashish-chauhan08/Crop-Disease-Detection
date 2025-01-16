# Plant Disease Prediction using CNN

This project is a plant disease prediction web app built using Flask and a Convolutional Neural Network (CNN) model. The model classifies plant diseases based on images uploaded by users. It helps in identifying diseases in various crops like Apple, Tomato, and more.

## Features

- Upload plant images to get predictions on potential diseases.
- Displays the predicted disease along with confidence.
- Lists supported plants and diseases for easy reference.
- Simple and intuitive user interface.
- Easy to deploy and run locally.

## Technologies Used

- **Flask**: A lightweight Python web framework used to create the API for serving predictions.
- **TensorFlow**: A deep learning framework used to build and train the CNN model.
- **Keras**: High-level API used to define the neural network architecture.
- **HTML/CSS/JS**: Frontend technologies for building the website interface.
- **Gunicorn**: WSGI HTTP Server used to deploy the app in production on Render.

## Installation

To run the project locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/plant-disease-prediction.git
   cd plant-disease-prediction

2.	Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask app:
   ```bash
   flask run
   ```

The app will be available at http://127.0.0.1:5000.


## Usage
	1.	Homepage: Upon visiting the homepage, you’ll find a simple user interface where you can upload images of plants.
	2.	Uploading an image: Click on the “Choose File” button to select an image from your device. Ensure that the image is clear and captures the affected part of the plant.
	3.	Prediction: Once the image is uploaded, the model processes it and predicts the disease along with a confidence score. The result will be displayed below the upload section.
	4.	View Supported Plants and Diseases: You can click on the “View Supported Plants and Diseases” button to see a list of all supported plants and diseases.
	5.	Predicted Output: The app will display the predicted disease in a clear and bold font for better visibility.


## Supported Plants and Diseases
The following plants and diseases are supported by the model:
	•	Apple:
	•	Apple scab
	•	Black rot
	•	Cedar apple rust
	•	Healthy
	•	Tomato:
	•	Bacterial spot
	•	Early blight
	•	Late blight
	•	Healthy
	•	Grape:
	•	Black rot
	•	Esca (Black Measles)
	•	Leaf blight (Isariopsis Leaf Spot)
	•	Healthy
	•	Peach:
	•	Bacterial spot
	•	Healthy
	•	Potato:
	•	Early blight
	•	Late blight
	•	Healthy
	•	Corn (Maize):
	•	Cercospora leaf spot Gray leaf spot
	•	Common rust
	•	Northern Leaf Blight
	•	Healthy
	•	Cherry:
	•	Powdery mildew
	•	Healthy
	•	Pepper:
	•	Bacterial spot
	•	Healthy
	•	Strawberry:
	•	Leaf scorch
	•	Healthy
	•	Others: Additional crops like Blueberry, Orange, Raspberry, and Soybean are also supported.


## Deployment

This app is hosted on Render. You can access the live version of the application at:

(To be available soon)
