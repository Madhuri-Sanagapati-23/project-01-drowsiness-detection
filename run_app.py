#!/usr/bin/env python3
"""
Clean launcher for Drowsiness Detection System
Suppresses TensorFlow warnings and provides better error handling
"""

import os
import sys
import warnings

def suppress_tensorflow_warnings():
    """Suppress TensorFlow and other warnings for cleaner output"""
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Suppress specific warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # Suppress OpenCV warnings
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

def check_environment():
    """Check if we're in the right environment"""
    print("🔍 Checking environment...")
    
    # Check if we're in virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Not running in virtual environment")
        print("   Recommendation: Activate virtual environment first")
        response = input("   Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("   Exiting...")
            sys.exit(1)
    
    # Check required files
    required_files = [
        "shape_predictor_68_face_landmarks.dat",
        "alarm.wav",
        "main.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all required files are present.")
        sys.exit(1)
    
    print("✅ Environment check passed")

def main():
    """Main launcher function"""
    print("🚀 Drowsiness Detection System Launcher")
    print("=" * 50)
    
    # Suppress warnings
    suppress_tensorflow_warnings()
    
    # Check environment
    check_environment()
    
    print("\n🎯 Starting Drowsiness Detection System...")
    print("   Press 'q' to quit, 's' to save screenshot")
    print("   Make sure your webcam is connected and accessible")
    print("-" * 50)
    
    try:
        # Import and run main application
        from main import main as run_main
        run_main()
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure webcam is connected and not in use by another application")
        print("2. Check if camera drivers are properly installed")
        print("3. Try running: python setup_check.py")
        print("4. Verify virtual environment is activated")
    finally:
        print("\n👋 Drowsiness Detection System stopped")

if __name__ == "__main__":
    main() 