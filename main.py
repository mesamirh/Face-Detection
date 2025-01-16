import cv2
import numpy as np
import time
import os
import logging
from datetime import datetime
import math
from scipy.spatial import distance as dist

class AdvancedFaceDetection:
    def __init__(self):
        logging.basicConfig(filename='face_detection.log', level=logging.INFO)
        
        try:
            # Core cascades
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            
            # Initialize camera
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, 60)
            
            # Unique tracking parameters
            self.golden_ratio = 1.618
            self.depth_reference = 50
            self.personal_space = 100
            self.face_history = {}
            self.movement_vectors = {}
            self.focus_map = np.zeros((1080, 1920))
            
            # Analytics storage
            self.metrics = {
                'orientation_history': [],
                'symmetry_scores': [],
                'proximity_alerts': [],
                'focus_times': {},
                'golden_ratios': [],
                'depth_estimates': []
            }
            
            # Create analytics directory
            os.makedirs("face_analytics", exist_ok=True)
            
        except Exception as e:
            logging.error(f"Initialization error: {str(e)}")
            raise

    def calculate_face_orientation(self, face_points):
        """Calculate face orientation in 3D space"""
        x, y, w, h = face_points
        center = (x + w//2, y + h//2)
        orientation = math.degrees(math.atan2(center[1] - 540, center[0] - 960))
        return orientation

    def estimate_depth(self, face_width):
        """Estimate face distance using width"""
        return (self.depth_reference * 1000) / face_width

    def analyze_golden_ratio(self, face_roi):
        """Analyze face proportions against golden ratio"""
        height, width = face_roi.shape[:2]
        ratio = width / height
        return abs(ratio - self.golden_ratio)

    def calculate_symmetry(self, face_roi):
        """Calculate face symmetry score"""
        height, width = face_roi.shape[:2]
        mid = width // 2
        left = face_roi[:, :mid]
        right = cv2.flip(face_roi[:, mid:], 1)
        return cv2.matchTemplate(left, right, cv2.TM_CCOEFF_NORMED)[0][0]

    def draw_ar_grid(self, frame, face_points):
        """Draw augmented reality grid on face"""
        x, y, w, h = face_points
        grid_size = 5
        cell_w, cell_h = w // grid_size, h // grid_size
        
        for i in range(grid_size + 1):
            cv2.line(frame, (x + i*cell_w, y), (x + i*cell_w, y + h), (0, 255, 255), 1)
            cv2.line(frame, (x, y + i*cell_h), (x + w, y + i*cell_h), (0, 255, 255), 1)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            face_id = f"{x}_{y}"
            
            # Calculate unique metrics
            orientation = self.calculate_face_orientation((x, y, w, h))
            depth = self.estimate_depth(w)
            symmetry = self.calculate_symmetry(face_roi)
            golden_score = self.analyze_golden_ratio(face_roi)
            
            # Update focus heat map
            self.focus_map[y:y+h, x:x+w] += 1
            
            # Draw visualizations
            self.draw_ar_grid(frame, (x, y, w, h))
            
            # Draw analytics
            cv2.putText(frame, f"Depth: {int(depth)}cm", (x, y-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(frame, f"Orient: {int(orientation)}°", (x, y-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(frame, f"Sym: {symmetry:.2f}", (x, y-60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Personal space warning
            if depth < self.personal_space:
                cv2.putText(frame, "PERSONAL SPACE!", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Store metrics
            self.metrics['orientation_history'].append(orientation)
            self.metrics['symmetry_scores'].append(symmetry)
            self.metrics['depth_estimates'].append(depth)
            self.metrics['golden_ratios'].append(golden_score)
        
        return frame

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            processed_frame = self.process_frame(frame)
            cv2.imshow('Unique Face Detection', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = AdvancedFaceDetection()
    detector.run()