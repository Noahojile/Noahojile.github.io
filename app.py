from flask import Flask, request, jsonify, render_template
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load TensorFlow Lite model
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class labels (adjust according to your dataset)
class_labels = ['Bean_Rust', 'Angular_Leaf_Spot', 'Healthy']

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the file temporarily
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    # Load and preprocess the image
    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get prediction result
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = class_labels[np.argmax(predictions)]

    return jsonify({'class': predicted_class})

# Serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
