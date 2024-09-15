from flask import Flask, request, jsonify
from PIL import Image
import io
from transformers import pipeline

# Initialize the pipeline for plant disease detection
pipe = pipeline("image-classification", model="susnato/plant_disease_detection-beans")

# Initialize Flask app
app = Flask(__name__)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    # Get the uploaded image
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))

    # Run the image classification
    result = pipe(image)

    # Return the result as JSON
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
