from flask import Flask, request, render_template, Response, url_for
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from datetime import datetime

app = Flask(__name__)

# Create directories for storing images
UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULT_FOLDER = os.path.join('static', 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Custom functions for model loading
def dice_coef(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred, smooth=100):
    return -dice_coef(y_true, y_pred, smooth)

def iou_coef(y_true, y_pred, smooth=100):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true + y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

# Load model with custom objects
model = load_model(
    r'C:\Users\91981\Downloads\BTS\unet_final.keras',
    custom_objects={
        'dice_coef': dice_coef,
        'dice_loss': dice_loss,
        'iou_coef': iou_coef
    }
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
        
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    try:
        # Save original image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        orig_filename = f"input_{timestamp}.png"
        result_filename = f"result_{timestamp}.png"
        
        orig_filepath = os.path.join(UPLOAD_FOLDER, orig_filename)
        result_filepath = os.path.join(RESULT_FOLDER, result_filename)
        
        # Save uploaded image
        file.save(orig_filepath)
        
        # Read and preprocess image for model
        img = cv2.imread(orig_filepath)
        img = cv2.resize(img, (256, 256))
        img_normalized = img / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        # Make prediction
        prediction = model.predict(img_batch)
        pred_binary = (prediction > 0.5).astype(np.uint8)
        
        # Save the result
        result_image = np.squeeze(pred_binary) * 255
        cv2.imwrite(result_filepath, result_image)
        
        # Return the image directly (binary format)
        with open(result_filepath, 'rb') as f:
            result_data = f.read()
        return Response(result_data, mimetype='image/png')

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return f"Error processing image: {str(e)}", 500

if __name__ == '__main__': 
    app.run(debug=True)
