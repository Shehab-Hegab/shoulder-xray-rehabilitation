# predictor.py (Enhanced Final Version)

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from datetime import datetime

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
            self.img_size = (256, 256)  # Must match the training image size
            print("Model loaded successfully.")
        except Exception as e:
            self.model = None
            print(f"Error loading model: {e}")

        # This dictionary maps the integer output of the model to the class name
        # The order depends on how ImageDataGenerator found the folders (alphabetical)
        self.class_names = {0: 'A1', 1: 'C1', 2: 'D1', 3: 'Others'}

        # Define conditions and recommendations based on the classified Glenoid Type
        self.recommendations = {
            'A1': {
                'condition': 'Normal Glenoid or minimal wear. No significant deformity.', 
                'prosthetic': 'Anatomic total shoulder arthroplasty (TSA) is typically recommended.',
                'details': 'The glenoid shows minimal erosion with maintained version. The humeral head is centered.'
            },
            'C1': {
                'condition': 'Dysplastic Glenoid with backward tilt (>25°).', 
                'prosthetic': 'An augmented implant or bone graft may be necessary to correct the tilt.',
                'details': 'Significant glenoid retroversion present. Consider preoperative CT for planning.'
            },
            'D1': {
                'condition': 'Posterior decentering of the humeral head on the Glenoid.', 
                'prosthetic': 'Reverse total shoulder arthroplasty (rTSA) is often the preferred solution.',
                'details': 'Posterior subluxation of the humeral head with asymmetric glenoid wear.'
            },
            'Others': {
                'condition': 'Complex cases (e.g., B2, B3) with bone loss or other factors.', 
                'prosthetic': 'Requires detailed orthopedic analysis for a specific recommendation.',
                'details': 'Complex pathology requiring individualized treatment planning.'
            }
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
            processed_image = self.preprocess_image(image_path)
            prediction = self.model.predict(processed_image)
            
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction) * 100
            
            glenoid_type = self.class_names.get(predicted_class_index, "Unknown")
            result_info = self.recommendations.get(glenoid_type, {
                'condition': 'Unknown', 
                'prosthetic': 'N/A',
                'details': 'Unable to determine pathology'
            })
            
            result_info['confidence'] = f"{confidence:.2f}%"
            result_info['glenoid_type'] = glenoid_type
            result_info['timestamp'] = datetime.now().isoformat()
            
            return result_info
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {'error': 'Failed to process image.'}