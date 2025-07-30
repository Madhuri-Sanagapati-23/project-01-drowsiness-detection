import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from config import MODEL_PATH

class EyeStateModel:
    def __init__(self):
        """Initialize the eye state classification model"""
        self.model = None
        self.is_trained = False
        
    def build_model(self, input_shape=(64, 64, 1), num_classes=2):
        """
        Build a CNN model for eye state classification
        
        Args:
            input_shape: Shape of input images
            num_classes: Number of classes (open/closed)
            
        Returns:
            tensorflow.keras.Model: Compiled CNN model
        """
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def load_model(self):
        """Load pre-trained model from file"""
        if os.path.exists(MODEL_PATH):
            try:
                self.model = tf.keras.models.load_model(MODEL_PATH)
                self.is_trained = True
                print(f"Model loaded successfully from {MODEL_PATH}")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        else:
            print(f"Model file not found at {MODEL_PATH}")
            return False
    
    def save_model(self):
        """Save the trained model to file"""
        if self.model is not None:
            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            
            try:
                self.model.save(MODEL_PATH)
                print(f"Model saved successfully to {MODEL_PATH}")
                return True
            except Exception as e:
                print(f"Error saving model: {e}")
                return False
        return False
    
    def predict(self, eye_image):
        """
        Predict eye state (open/closed)
        
        Args:
            eye_image: Preprocessed eye image
            
        Returns:
            tuple: (prediction, confidence)
        """
        if self.model is None or not self.is_trained:
            return None, 0.0
        
        try:
            prediction = self.model.predict(eye_image, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            # 0: closed, 1: open
            return predicted_class, confidence
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None, 0.0
    
    def train_model(self, train_data, train_labels, validation_data=None, epochs=50, batch_size=32):
        """
        Train the eye state classification model
        
        Args:
            train_data: Training images
            train_labels: Training labels
            validation_data: Validation data (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if self.model is None:
            self.model = self.build_model()
        
        # Convert labels to categorical
        from tensorflow.keras.utils import to_categorical
        train_labels_cat = to_categorical(train_labels, num_classes=2)
        
        # Training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            tf.keras.callbacks.ModelCheckpoint(
                MODEL_PATH, 
                save_best_only=True, 
                monitor='val_accuracy'
            )
        ]
        
        # Train the model
        history = self.model.fit(
            train_data,
            train_labels_cat,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        return history

def create_sample_dataset():
    """
    Create a sample dataset for training (placeholder function)
    In a real scenario, you would collect actual eye images
    """
    print("Sample dataset creation function - implement with real data collection")
    print("You would need to:")
    print("1. Collect images of open and closed eyes")
    print("2. Preprocess them using preprocess_eye_image function")
    print("3. Organize them into train/validation sets")
    print("4. Use the train_model method to train the CNN")
    
    # Placeholder for demonstration
    sample_data = np.random.random((100, 64, 64, 1))
    sample_labels = np.random.randint(0, 2, 100)
    
    return sample_data, sample_labels