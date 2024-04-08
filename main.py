from flask import Flask, render_template, request
import numpy as np
import cv2
import tensorflow as tf  # Assuming you're using TensorFlow for your model
# Import your pre-trained neural network model here (replace with your model name)

app = Flask(__name__)

# Load your trained model (replace with your loading logic)
model = tf.keras.models.load_model('catDogModel.keras')

@app.route('/')
def index():
    return "hello worlds"

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    # Preprocess image (resize, normalize, etc.)
    image_data = preprocess_image(image_file.read())
    # Add batch dimension
    prediction = model.predict(image_data)
    # Process prediction results
    predicted_class = np.argmax(prediction)
    if predicted_class==1:
        prediction_pet = "Cat"
    elif predicted_class==0:
        prediction_pet = "Dog"
    confidence = prediction[0][predicted_class]  # Assuming prediction is a probability distribution

    # Return a dictionary for the response
    return {'class': predicted_class, 'confidence': prediction_pet}
# Function to preprocess image (replace with your logic)
def preprocess_image(image_bytes):
    # Resize, convert to grayscale, normalize pixel values, etc.
    image = cv2.imread(image_bytes, cv2.IMREAD_GRAYSCALE)
    image_array = cv2.resize(image, (50, 50))
    image_array = np.array(image_array).reshape(-1, 50, 50, 1)
    return image_array

# Function to process prediction results (replace with your logic)
def process_prediction(prediction):
    # Get the class label with the highest probability
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class

if __name__ == '__main__':
    app.run(debug=True)
