#!/usr/bin/env python3
"""
Demo Mode for Drowsiness Detection System
Simulates the system without requiring a camera
Useful for testing and demonstration purposes
"""

import cv2
import numpy as np
import time
import threading
import winsound
import os
import sys
import warnings

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

from utils import DrowsinessDetector
from config import *

class DrowsinessDetectionDemo:
    def __init__(self):
        """Initialize the demo mode"""
        self.detector = DrowsinessDetector()
        self.is_running = False
        self.alert_playing = False
        self.demo_frame = None
        self.simulation_time = 0
        
        # Create a demo frame
        self.create_demo_frame()
        
    def create_demo_frame(self):
        """Create a simulated video frame"""
        # Create a black frame
        self.demo_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some visual elements
        cv2.putText(self.demo_frame, "DROWSINESS DETECTION DEMO", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(self.demo_frame, "Simulated Mode - No Camera Required", (50, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw a simulated face outline
        cv2.ellipse(self.demo_frame, (320, 240), (120, 160), 0, 0, 360, (100, 100, 100), 2)
        
        # Draw simulated eyes
        cv2.circle(self.demo_frame, (280, 200), 25, (150, 150, 150), 2)  # Left eye
        cv2.circle(self.demo_frame, (360, 200), 25, (150, 150, 150), 2)  # Right eye
        
    def simulate_ear_values(self):
        """Simulate realistic EAR values over time"""
        # Create a pattern that simulates normal blinking and drowsiness
        base_time = time.time()
        
        # Normal blinking pattern (every 3-5 seconds)
        blink_cycle = int(base_time) % 4
        
        if blink_cycle == 0:
            # Normal state
            return 0.25, 0.26
        elif blink_cycle == 1:
            # Slight drowsiness
            return 0.20, 0.21
        elif blink_cycle == 2:
            # More drowsy
            return 0.18, 0.19
        else:
            # Very drowsy (should trigger alert)
            return 0.15, 0.16
    
    def update_demo_frame(self, left_ear, right_ear, status, drowsy_time=None):
        """Update the demo frame with current information"""
        # Clear previous text
        cv2.rectangle(self.demo_frame, (10, 100), (630, 450), (0, 0, 0), -1)
        
        # Redraw face and eyes
        cv2.ellipse(self.demo_frame, (320, 240), (120, 160), 0, 0, 360, (100, 100, 100), 2)
        cv2.circle(self.demo_frame, (280, 200), 25, (150, 150, 150), 2)
        cv2.circle(self.demo_frame, (360, 200), 25, (150, 150, 150), 2)
        
        # Update status
        status_color = COLORS['GREEN'] if status == "NORMAL" else COLORS['YELLOW'] if status == "DROWSY" else COLORS['RED']
        cv2.putText(self.demo_frame, f"Status: {status}", (50, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Show EAR values
        cv2.putText(self.demo_frame, f"Left EAR: {left_ear:.3f}", (50, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['WHITE'], 1)
        cv2.putText(self.demo_frame, f"Right EAR: {right_ear:.3f}", (50, 170),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['WHITE'], 1)
        
        # Show drowsy time if applicable
        if drowsy_time is not None:
            cv2.putText(self.demo_frame, f"Drowsy Time: {drowsy_time:.1f}s", (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['YELLOW'], 1)
        
        # Show simulation time
        cv2.putText(self.demo_frame, f"Simulation Time: {self.simulation_time:.1f}s", (50, 230),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['WHITE'], 1)
        
        # Add instructions
        cv2.putText(self.demo_frame, "Press 'q' to quit, 's' to save screenshot", (50, 260),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['WHITE'], 1)
        cv2.putText(self.demo_frame, "This is a simulation - no real camera required", (50, 280),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['WHITE'], 1)
        
        # Visual indicators for eye state
        eye_color = COLORS['GREEN'] if status == "NORMAL" else COLORS['RED']
        cv2.circle(self.demo_frame, (280, 200), 20, eye_color, -1)  # Left eye
        cv2.circle(self.demo_frame, (360, 200), 20, eye_color, -1)  # Right eye
        
    def play_alarm(self):
        """Play alarm sound"""
        if not self.alert_playing and os.path.exists(ALARM_FILE):
            self.alert_playing = True
            try:
                winsound.PlaySound(ALARM_FILE, winsound.SND_FILENAME | winsound.SND_ASYNC)
                threading.Timer(ALARM_DURATION, self.stop_alarm).start()
            except Exception as e:
                print(f"Error playing alarm: {e}")
                self.alert_playing = False
    
    def stop_alarm(self):
        """Stop the alarm"""
        self.alert_playing = False
    
    def run(self):
        """Run the demo mode"""
        print("🎭 Drowsiness Detection Demo Mode")
        print("Simulating drowsiness detection without camera")
        print("Press 'q' to quit, 's' to save screenshot")
        print("-" * 50)
        
        self.is_running = True
        start_time = time.time()
        
        while self.is_running:
            # Update simulation time
            self.simulation_time = time.time() - start_time
            
            # Simulate EAR values
            left_ear, right_ear = self.simulate_ear_values()
            ear = (left_ear + right_ear) / 2.0
            
            # Determine drowsiness
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
            
            # Update demo frame
            self.update_demo_frame(left_ear, right_ear, status, drowsy_time if is_drowsy else None)
            
            # Trigger alarm if needed
            if should_alert:
                self.play_alarm()
            
            # Display the frame
            cv2.imshow("Drowsiness Detection Demo", self.demo_frame)
            
            # Handle key presses
            key = cv2.waitKey(100) & 0xFF  # 10 FPS
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"demo_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, self.demo_frame)
                print(f"Screenshot saved: {filename}")
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        cv2.destroyAllWindows()
        print("\n🎭 Demo mode stopped")

def main():
    """Main function to run the demo"""
    try:
        demo = DrowsinessDetectionDemo()
        demo.run()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo error: {e}")

if __name__ == "__main__":
    main() 