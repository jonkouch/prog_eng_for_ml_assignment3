import joblib
import numpy as np
from PIL import Image
import os

# Load the trained model
model = joblib.load('digits_model.joblib')

# Directory containing the images to be classified
input_dir = '/input_images'
output_dir = '/output_predictions'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to preprocess image and make prediction
def classify_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((8, 8), Image.ANTIALIAS)
    img_data = np.array(img).reshape(1, -1)
    prediction = model.predict(img_data)[0]
    return prediction

# Process each image file in the input directory
for image_file in os.listdir(input_dir):
    if image_file.endswith('.png'):
        image_path = os.path.join(input_dir, image_file)
        prediction = classify_image(image_path)
        output_file = os.path.join(output_dir, f"{image_file}.txt")
        with open(output_file, 'w') as f:
            f.write(str(prediction))
