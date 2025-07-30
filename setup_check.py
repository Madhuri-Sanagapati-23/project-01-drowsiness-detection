#!/usr/bin/env python3
"""
Setup verification script for drowsiness detection project
Helps users verify their environment is properly configured
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("   ✅ Python version is compatible")
        return True
    else:
        print("   ❌ Python 3.8+ is required")
        return False

def check_virtual_environment():
    """Check if running in virtual environment"""
    print("\n🔧 Checking virtual environment...")
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("   ✅ Running in virtual environment")
        print(f"   Virtual env: {sys.prefix}")
        return True
    else:
        print("   ⚠️  Not running in virtual environment")
        print("   Recommendation: Use virtual environment for isolation")
        return False

def check_dependencies():
    """Check if all dependencies are installed"""
    print("\n📦 Checking dependencies...")
    
    dependencies = [
        ('opencv-python', 'cv2'),
        ('tensorflow', 'tensorflow'),
        ('dlib', 'dlib'),
        ('imutils', 'imutils'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('pillow', 'PIL')
    ]
    
    all_installed = True
    for package_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name} - Not installed")
            all_installed = False
    
    return all_installed

def check_required_files():
    """Check if required files exist"""
    print("\n📁 Checking required files...")
    
    required_files = [
        ("shape_predictor_68_face_landmarks.dat", "dlib facial landmark model"),
        ("alarm.wav", "audio alert file"),
        ("config.py", "configuration file"),
        ("utils.py", "utility functions"),
        ("model.py", "CNN model definition"),
        ("main.py", "main application")
    ]
    
    all_exist = True
    for filename, description in required_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"   ✅ {filename} ({size:,} bytes) - {description}")
        else:
            print(f"   ❌ {filename} - Missing ({description})")
            all_exist = False
    
    return all_exist

def check_camera_access():
    """Check if camera is accessible"""
    print("\n📹 Checking camera access...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("   ✅ Camera is accessible")
            cap.release()
            return True
        else:
            print("   ❌ Camera not accessible")
            return False
    except Exception as e:
        print(f"   ❌ Camera check failed: {e}")
        return False

def check_audio_support():
    """Check audio support"""
    print("\n🔊 Checking audio support...")
    try:
        import winsound
        print("   ✅ Audio support available (winsound)")
        return True
    except ImportError:
        print("   ❌ Audio support not available")
        return False

def check_system_info():
    """Display system information"""
    print("\n💻 System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Python: {platform.python_implementation()} {platform.python_version()}")

def main():
    """Run all checks"""
    print("🔍 Drowsiness Detection - Setup Verification")
    print("=" * 50)
    
    checks = [
        check_python_version,
        check_virtual_environment,
        check_dependencies,
        check_required_files,
        check_camera_access,
        check_audio_support
    ]
    
    passed = 0
    total = len(checks)
    
    for check in checks:
        try:
            if check():
                passed += 1
        except Exception as e:
            print(f"   ❌ Check failed with exception: {e}")
    
    check_system_info()
    
    print("\n" + "=" * 50)
    print(f"📊 Verification Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! Your environment is ready.")
        print("\n🚀 You can now run the application:")
        print("   python main.py")
    else:
        print("⚠️  Some checks failed. Please address the issues above.")
        print("\n🔧 Common solutions:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Download shape_predictor_68_face_landmarks.dat")
        print("3. Create alarm.wav file")
        print("4. Ensure camera is connected and accessible")
        print("5. Use virtual environment: python -m venv .venv")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 