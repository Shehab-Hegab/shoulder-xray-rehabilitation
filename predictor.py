# predictor.py (Final Version - English Comments)

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class XRayPredictor:
    """
    Handles loading the model and making predictions.
    This class separates the AI logic from the user interface.
    """
    def __init__(self, model_path):
        """
        Loads the TensorFlow/Keras model upon initialization.
        """
        try:
            print("Loading model...")
            self.model = tf.keras.models.load_model(model_path)
            self.img_size = (256, 256) # Must match the training image size
            print("Model loaded successfully.")
        except Exception as e:
            self.model = None
            print(f"Error loading model: {e}")

        # This dictionary maps the integer output of the model to the class name
        # The order depends on how ImageDataGenerator found the folders (alphabetical)
        self.class_names = {0: 'A1', 1: 'C1', 2: 'D1', 3: 'Others'}

        # Define conditions and recommendations based on the classified Glenoid Type
        self.recommendations = {
            'A1': {'condition': 'Normal or minimal wear', 'prosthetic': 'Anatomic total shoulder arthroplasty (TSA)'},
            'C1': {'condition': 'Backward tilt (>25°)', 'prosthetic': 'Augmented implant or bone graft'},
            'D1': {'condition': 'Posterior decentering', 'prosthetic': 'Reverse total shoulder arthroplasty (rTSA)'},
            'Others': {'condition': 'Varies (e.g., B2, B3)', 'prosthetic': 'Consult detailed orthopedic analysis'}
        }

    def preprocess_image(self, image_path):
        """
        Loads and preprocesses a single image for prediction.
        """
        # Load the image and resize it to the model's expected input size
        img = load_img(image_path, target_size=self.img_size, color_mode='rgb')
        # Convert the image to a NumPy array
        img_array = img_to_array(img)
        # Add a batch dimension (e.g., from (256, 256, 3) to (1, 256, 256, 3))
        img_array = np.expand_dims(img_array, axis=0)
        # No rescaling is done, to match the original training notebook's logic
        return img_array

    def predict(self, image_path):
        """
        Takes an image path, preprocesses it, and returns the prediction results.
        """
        if not self.model:
            return {'error': "Model not loaded"}
        
        try:
            # 1. Preprocess the image
            processed_image = self.preprocess_image(image_path)
            # 2. Get the model's prediction
            prediction = self.model.predict(processed_image)
            
            # 3. Find the highest probability class and its confidence
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction) * 100
            
            # 4. Map the results to human-readable information
            glenoid_type = self.class_names.get(predicted_class_index, "Unknown")
            result_info = self.recommendations.get(glenoid_type, {'condition': 'Unknown', 'prosthetic': 'N/A'})
            
            # 5. Add confidence and type to the result dictionary
            result_info['confidence'] = f"{confidence:.2f}%"
            result_info['glenoid_type'] = glenoid_type
            
            return result_info
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {'error': 'Failed to process image.'}