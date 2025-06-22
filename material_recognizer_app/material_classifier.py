# material_classifier.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
import os
from PIL import Image

from config import MODEL_PATH, IMG_HEIGHT, IMG_WIDTH

class MaterialClassifier:
    """
    Provides functionality to load a trained material recognition model
    and perform predictions on images or video streams.
    """
    def __init__(self, model_path=MODEL_PATH, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, class_names=None):
        """
        Initializes the classifier by loading the pre-trained model.

        Args:
            model_path (str): Path to the saved Keras model (.h5).
            img_height (int): Height the input images should be resized to.
            img_width (int): Width the input images should be resized to.
            class_names (list): List of material class names, in the order they were trained.
                                If None, you must set them manually or load from a source.
        """
        self.model_path = model_path
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.class_names = class_names

        self._load_model()

    def _load_model(self):
        """Loads the Keras model from the specified path."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}. Please train the model first.")
        try:
            self.model = load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            raise Exception(f"Error loading model from {self.model_path}: {e}")

    def set_class_names(self, class_names):
        """Sets the class names for the classifier."""
        if not isinstance(class_names, list) or not all(isinstance(c, str) for c in class_names):
            raise ValueError("class_names must be a list of strings.")
        self.class_names = class_names
        print(f"Class names set: {self.class_names}")

    def preprocess_image(self, img):
        """
        Preprocesses a raw image (PIL Image or OpenCV frame) for model prediction.

        Args:
            img: A PIL Image object or a NumPy array (OpenCV frame).

        Returns:
            np.ndarray: Preprocessed image array suitable for model input.
        """
        if isinstance(img, np.ndarray): # If it's an OpenCV frame (BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
            img = Image.fromarray(img) # Convert to PIL Image for resize and array conversion
        elif not isinstance(img, Image.Image):
            raise TypeError("Input must be a PIL Image or OpenCV NumPy array.")

        img = img.resize((self.img_width, self.img_height))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        img_array /= 255.0 # Normalize to [0, 1]
        return img_array

    def predict_image(self, image_input):
        """
        Predicts the material type from a single image.

        Args:
            image_input (str or PIL.Image.Image or np.ndarray):
                Path to the image file, a PIL Image object, or an OpenCV frame (numpy array).

        Returns:
            tuple: (predicted_material_name, confidence_score)
        Raises:
            ValueError: If class names are not set.
        """
        if self.class_names is None or not self.class_names:
            raise ValueError("Class names are not set. Call set_class_names() first.")
        if self.model is None:
            raise ValueError("Model not loaded. Check model_path.")

        if isinstance(image_input, str):
            try:
                img = image.load_img(image_input)
            except Exception as e:
                raise FileNotFoundError(f"Could not load image from path '{image_input}': {e}")
        elif isinstance(image_input, (Image.Image, np.ndarray)):
            img = image_input
        else:
            raise TypeError("image_input must be a file path, PIL Image, or OpenCV NumPy array.")

        processed_img = self.preprocess_image(img)
        predictions = self.model.predict(processed_img)[0]
        predicted_class_index = np.argmax(predictions)
        predicted_material = self.class_names[predicted_class_index]
        confidence = predictions[predicted_class_index]

        return predicted_material, float(confidence)

    def process_video_stream(self, video_source=0):
        """
        Processes a live video stream (e.g., webcam) or a video file
        and displays real-time material predictions.

        Args:
            video_source (int or str): 0 for default webcam, or path to a video file.
        Raises:
            ValueError: If class names are not set.
            RuntimeError: If video stream cannot be opened.
        """
        if self.class_names is None or not self.class_names:
            raise ValueError("Class names are not set. Call set_class_names() first.")
        if self.model is None:
            raise ValueError("Model not loaded. Check model_path.")

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Error: Could not open video source {video_source}.")

        print(f"Processing video from source: {video_source}. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or error reading frame.")
                break

            try:
                # Direct prediction on OpenCV frame (NumPy array)
                predicted_material, confidence = self.predict_image(frame)

                # Display prediction on the frame
                text = f"Material: {predicted_material} ({confidence:.2f})"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.imshow('Material Recognition Stream', frame)
            except Exception as e:
                print(f"Error during video frame processing: {e}")
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Video stream closed.")

# Example Usage (for testing material_classifier independently)
if __name__ == "__main__":
    print("Testing MaterialClassifier...")
    # NOTE: For this to work, you MUST have a trained model saved at config.MODEL_PATH
    # and a 'data' directory with the same classes that were used for training.

    # 1. Create a dummy model (if you don't have a real one trained yet)
    # This is just to allow the classifier to load something.
    # In a real scenario, you'd run model_trainer.py first.
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"No model found at {MODEL_PATH}. Creating a dummy model for testing...")
            dummy_model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(2, activation='softmax') # Assuming 2 dummy classes
            ])
            dummy_model.compile(optimizer='adam', loss='categorical_crossentropy')
            dummy_model.save(MODEL_PATH)
            print("Dummy model created. Please replace with a real trained model for accurate results.")
        else:
            print("Found existing model. Using it for testing.")

        # 2. Define dummy class names (must match what was used for training the model)
        # In a real application, you'd get these from your data_loader or a config file.
        dummy_class_names = ['class_a', 'class_b'] # Example dummy classes

        classifier = MaterialClassifier(class_names=dummy_class_names)

        # Test single image prediction
        # Create a dummy image file
        from PIL import Image
        dummy_img_path = 'test_image_for_classifier.jpg'
        Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color = 'blue').save(dummy_img_path)

        print("\nTesting single image prediction:")
        predicted_mat, conf = classifier.predict_image(dummy_img_path)
        print(f"Predicted: {predicted_mat}, Confidence: {conf:.2f}")

        os.remove(dummy_img_path) # Clean up dummy image

        # Test video stream prediction (requires webcam or video file)
        # Uncomment the following line to test video stream
        # print("\nTesting video stream (press 'q' to quit):")
        # classifier.process_video_stream(0) # Use 0 for webcam, or path to video file

        print("\nMaterialClassifier test complete.")

    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure you have a trained model and correct paths.")
    except Exception as e:
        print(f"An unexpected error occurred during classifier test: {e}")
    finally:
        # Clean up dummy model if created for testing
        if os.path.exists(MODEL_PATH) and "dummy" in MODEL_PATH: # crude check
            # This logic needs to be more robust if you're seriously managing dummy models
            # For now, it's just a hint. A better practice would be to use a separate test model path.
            print(f"Removing dummy model at {MODEL_PATH} if it was just created for this test.")
            # os.remove(MODEL_PATH) # Uncomment if you want to aggressively clean up