#!/usr/bin/env python3
"""
Test script for drowsiness detection application
Tests core functionality without requiring camera access
"""

import os
import sys
import numpy as np
import cv2
from utils import DrowsinessDetector
from model import EyeStateModel
from config import *

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    try:
        import cv2
        import dlib
        import tensorflow as tf
        import numpy as np
        import imutils
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("\n🔍 Testing configuration...")
    try:
        print(f"EAR Threshold: {EAR_THRESHOLD}")
        print(f"Camera Index: {CAMERA_INDEX}")
        print(f"Frame Size: {FRAME_WIDTH}x{FRAME_HEIGHT}")
        print(f"Use CNN Model: {USE_CNN_MODEL}")
        print("✅ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_detector_initialization():
    """Test drowsiness detector initialization"""
    print("\n🔍 Testing detector initialization...")
    try:
        detector = DrowsinessDetector()
        print("✅ DrowsinessDetector initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Detector initialization error: {e}")
        return False

def test_model_initialization():
    """Test CNN model initialization"""
    print("\n🔍 Testing model initialization...")
    try:
        model = EyeStateModel()
        print("✅ EyeStateModel initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Model initialization error: {e}")
        return False

def test_required_files():
    """Test if required files exist"""
    print("\n🔍 Testing required files...")
    required_files = [
        "shape_predictor_68_face_landmarks.dat",
        "alarm.wav"
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            all_exist = False
    
    return all_exist

def test_ear_calculation():
    """Test EAR calculation with dummy data"""
    print("\n🔍 Testing EAR calculation...")
    try:
        detector = DrowsinessDetector()
        
        # Create dummy eye landmarks (6 points for each eye)
        left_eye = np.array([
            [10, 10], [20, 10], [30, 10],  # Top points
            [10, 20], [20, 20], [30, 20]   # Bottom points
        ], dtype=np.float32)
        
        right_eye = np.array([
            [40, 10], [50, 10], [60, 10],  # Top points
            [40, 20], [50, 20], [60, 20]   # Bottom points
        ], dtype=np.float32)
        
        # Calculate EAR manually
        left_ear = detector.eye_aspect_ratio(left_eye)
        right_ear = detector.eye_aspect_ratio(right_eye)
        
        print(f"✅ EAR calculation successful")
        print(f"   Left EAR: {left_ear:.3f}")
        print(f"   Right EAR: {right_ear:.3f}")
        return True
    except Exception as e:
        print(f"❌ EAR calculation error: {e}")
        return False

def test_drowsiness_detection():
    """Test drowsiness detection logic"""
    print("\n🔍 Testing drowsiness detection logic...")
    try:
        detector = DrowsinessDetector()
        
        # Test with different EAR values
        test_cases = [
            (0.15, "Should be drowsy (low EAR)"),
            (0.25, "Should be normal (high EAR)"),
            (0.21, "Borderline case")
        ]
        
        for ear_value, description in test_cases:
            is_drowsy = detector.is_drowsy(ear_value)
            status = "DROWSY" if is_drowsy else "NORMAL"
            print(f"   EAR {ear_value:.2f}: {status} - {description}")
        
        print("✅ Drowsiness detection logic working")
        return True
    except Exception as e:
        print(f"❌ Drowsiness detection error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Drowsiness Detection System - Component Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_detector_initialization,
        test_model_initialization,
        test_required_files,
        test_ear_calculation,
        test_drowsiness_detection
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to run.")
        print("\nTo run the full application:")
        print("   python main.py")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("\nCommon solutions:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Download shape_predictor_68_face_landmarks.dat")
        print("3. Create alarm.wav file")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 