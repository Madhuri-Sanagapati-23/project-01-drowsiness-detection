import cv2
import numpy as np
import dlib
from imutils import face_utils
from config import *

class DrowsinessDetector:
    def __init__(self):
        """Initialize the drowsiness detector with facial landmark predictor"""
        # Initialize dlib's face detector and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Extract indexes of facial landmarks for the left and right eye
        self.left_eye_start, self.left_eye_end = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self.right_eye_start, self.right_eye_end = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        
        # Initialize drowsy state tracking
        self.drowsy_start_time = None
        
    def eye_aspect_ratio(self, eye):
        """
        Calculate the Eye Aspect Ratio (EAR) for a given eye
        
        Args:
            eye: Array of 6 (x, y) coordinates representing the eye landmarks
            
        Returns:
            float: The calculated EAR value
        """
        # Compute the euclidean distances between the vertical eye landmarks
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        
        # Compute the euclidean distance between the horizontal eye landmarks
        C = np.linalg.norm(eye[0] - eye[3])
        
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        
        return ear
    
    def detect_eyes(self, frame):
        """
        Detect eyes in the frame and calculate EAR values
        
        Args:
            frame: Input video frame
            
        Returns:
            tuple: (left_ear, right_ear, left_eye_coords, right_eye_coords, face_landmarks)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = self.detector(gray, 0)
        
        if len(faces) == 0:
            return None, None, None, None, None
        
        # For simplicity, we'll use the first face detected
        face = faces[0]
        
        # Determine the facial landmarks for the face region
        shape = self.predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        # Extract the left and right eye coordinates
        left_eye = shape[self.left_eye_start:self.left_eye_end]
        right_eye = shape[self.right_eye_start:self.right_eye_end]
        
        # Calculate the eye aspect ratio for both eyes
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        
        # Average the eye aspect ratio together for both eyes
        ear = (left_ear + right_ear) / 2.0
        
        return left_ear, right_ear, left_eye, right_eye, shape
    
    def draw_eye_landmarks(self, frame, left_eye, right_eye, shape):
        """
        Draw eye landmarks and face outline on the frame
        
        Args:
            frame: Input video frame
            left_eye: Left eye coordinates
            right_eye: Right eye coordinates
            shape: All facial landmarks
        """
        if SHOW_LANDMARKS:
            # Draw the face outline
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, COLORS['BLUE'], -1)
            
            # Draw the left eye outline
            left_eye_hull = cv2.convexHull(left_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, COLORS['GREEN'], 2)
            
            # Draw the right eye outline
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [right_eye_hull], -1, COLORS['GREEN'], 2)
    
    def draw_status(self, frame, ear, status, drowsy_time=None):
        """
        Draw status information on the frame
        
        Args:
            frame: Input video frame
            ear: Current EAR value
            status: Current status ("ALERT", "DROWSY", "NORMAL")
            drowsy_time: Time spent in drowsy state
        """
        if SHOW_EAR_VALUE:
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['WHITE'], 2)
        
        if SHOW_STATUS:
            # Choose color based on status
            if status == "ALERT":
                color = COLORS['RED']
            elif status == "DROWSY":
                color = COLORS['YELLOW']
            else:
                color = COLORS['GREEN']
            
            cv2.putText(frame, f"Status: {status}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if drowsy_time is not None:
                cv2.putText(frame, f"Drowsy Time: {drowsy_time:.1f}s", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        

    
    def is_drowsy(self, ear):
        """
        Determine if the person is drowsy based on EAR value
        
        Args:
            ear: Current EAR value
            
        Returns:
            bool: True if drowsy, False otherwise
        """
        return ear < EAR_THRESHOLD
    
    def update_drowsy_state(self, is_drowsy, current_time):
        """
        Update drowsy state tracking
        
        Args:
            is_drowsy: Whether person is currently drowsy
            current_time: Current timestamp
            
        Returns:
            tuple: (should_alert, drowsy_time)
        """
        if is_drowsy:
            if self.drowsy_start_time is None:
                self.drowsy_start_time = current_time
            else:
                drowsy_time = current_time - self.drowsy_start_time
                if drowsy_time >= DROWSY_TIME_THRESHOLD:
                    return True, drowsy_time
        else:
            self.drowsy_start_time = None
        
        return False, 0.0

def preprocess_eye_image(eye_coords, frame, target_size=(64, 64)):
    """
    Preprocess eye image for CNN model input
    
    Args:
        eye_coords: Eye landmark coordinates
        frame: Input video frame
        target_size: Target size for the eye image
        
    Returns:
        numpy.ndarray: Preprocessed eye image
    """
    # Create a bounding box around the eye
    x_coords = [coord[0] for coord in eye_coords]
    y_coords = [coord[1] for coord in eye_coords]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Add padding
    padding = 10
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(frame.shape[1], x_max + padding)
    y_max = min(frame.shape[0], y_max + padding)
    
    # Extract eye region
    eye_region = frame[y_min:y_max, x_min:x_max]
    
    if eye_region.size == 0:
        return None
    
    # Convert to grayscale
    eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    
    # Resize to target size
    eye_resized = cv2.resize(eye_gray, target_size)
    
    # Normalize pixel values
    eye_normalized = eye_resized / 255.0
    
    # Reshape for model input
    eye_input = eye_normalized.reshape(1, target_size[0], target_size[1], 1)
    
    return eye_input 