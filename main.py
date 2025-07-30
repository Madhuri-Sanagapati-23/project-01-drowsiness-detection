import cv2
import time
import threading
import numpy as np
import winsound
import os
import sys
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from utils import DrowsinessDetector, preprocess_eye_image
from model import EyeStateModel
from config import *

class DrowsinessDetectionApp:
    def __init__(self):
        """Initialize the drowsiness detection application"""
        global USE_CNN_MODEL
        
        self.detector = DrowsinessDetector()
        self.eye_model = EyeStateModel()
        self.cap = None
        self.is_running = False
        self.alert_playing = False
        
        # Initialize model if using CNN
        if USE_CNN_MODEL:
            if not self.eye_model.load_model():
                print("Warning: CNN model not found. Falling back to EAR-based detection.")
                USE_CNN_MODEL = False
        
        # Check for required files
        self.check_requirements()
    
    def check_requirements(self):
        """Check if required files exist"""
        if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
            print("Missing required file: shape_predictor_68_face_landmarks.dat")
            print("\nPlease download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            print("Extract and place it in the project root directory")
            sys.exit(1)
    
    def play_alarm(self):
        """Play alarm sound in a separate thread"""
        if not self.alert_playing and os.path.exists(ALARM_FILE):
            self.alert_playing = True
            try:
                # Use winsound to play the alarm
                winsound.PlaySound(ALARM_FILE, winsound.SND_FILENAME | winsound.SND_ASYNC)
                # Stop alarm after duration
                threading.Timer(ALARM_DURATION, self.stop_alarm).start()
            except Exception as e:
                print(f"Error playing alarm: {e}")
                self.alert_playing = False
    
    def stop_alarm(self):
        """Stop the alarm"""
        self.alert_playing = False
    
    def process_frame(self, frame):
        """
        Process a single frame for drowsiness detection
        
        Args:
            frame: Input video frame
            
        Returns:
            tuple: (processed_frame, should_alert, status)
        """
        # Detect eyes and calculate EAR
        left_ear, right_ear, left_eye, right_eye, shape = self.detector.detect_eyes(frame)
        
        if left_ear is None:
            # No face detected
            cv2.putText(frame, "No Face Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['RED'], 2)
            return frame, False, "NO_FACE"
        
        # Calculate average EAR
        ear = (left_ear + right_ear) / 2.0
        
        # Determine drowsiness using EAR or CNN model
        if USE_CNN_MODEL:
            # Use CNN model for classification
            left_eye_input = preprocess_eye_image(left_eye, frame)
            right_eye_input = preprocess_eye_image(right_eye, frame)
            
            if left_eye_input is not None and right_eye_input is not None:
                left_pred, left_conf = self.eye_model.predict(left_eye_input)
                right_pred, right_conf = self.eye_model.predict(right_eye_input)
                
                # Average predictions from both eyes
                avg_pred = (left_pred + right_pred) / 2
                avg_conf = (left_conf + right_conf) / 2
                
                is_drowsy = avg_pred == 0  # 0: closed, 1: open
            else:
                is_drowsy = self.detector.is_drowsy(ear)
        else:
            # Use EAR-based detection
            is_drowsy = self.detector.is_drowsy(ear)
        
        # Update drowsy state tracking
        current_time = time.time()
        should_alert, drowsy_time = self.detector.update_drowsy_state(is_drowsy, current_time)
        
        # Determine status
        if should_alert:
            status = "ALERT"
        elif is_drowsy:
            status = "DROWSY"
        else:
            status = "NORMAL"
        
        # Draw visual elements
        self.detector.draw_eye_landmarks(frame, left_eye, right_eye, shape)
        self.detector.draw_status(frame, ear, status, drowsy_time if is_drowsy else None)
        
        # Draw EAR values for each eye
        if SHOW_EAR_VALUE:
            cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['WHITE'], 1)
            cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (10, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['WHITE'], 1)
        
        return frame, should_alert, status
    
    def run(self):
        """Main application loop"""
        # Initialize video capture
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Drowsiness Detection System Started")
        print("Press 'q' to quit, 's' to save screenshot")
        print(f"Detection Method: {'CNN Model' if USE_CNN_MODEL else 'EAR-based'}")
        print("-" * 50)
        
        self.is_running = True
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process the frame
            processed_frame, should_alert, status = self.process_frame(frame)
            
            # Trigger alarm if needed
            if should_alert:
                self.play_alarm()
            
            # Display the frame
            cv2.imshow("Drowsiness Detection", processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"Screenshot saved: {filename}")
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\nDrowsiness Detection System Stopped")

def main():
    """Main function to run the drowsiness detection application"""
    try:
        app = DrowsinessDetectionApp()
        app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()