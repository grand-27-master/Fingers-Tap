import cv2
import mediapipe as mp
import time
import numpy as np
from collections import defaultdict

class FingerTapDetector:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Parameters
        self.DISTANCE_THRESHOLD = 40  
        self.COOLDOWN_PERIOD = 1.0    
        
        # Finger landmarks for each finger
        self.FINGER_LANDMARKS = {
            'INDEX': self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            'MIDDLE': self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            'RING': self.mp_hands.HandLandmark.RING_FINGER_TIP,
            'PINKY': self.mp_hands.HandLandmark.PINKY_TIP
        }
        
        # Initialize tracking variables
        self.last_tap_times = defaultdict(float)
        self.tap_counts = defaultdict(int)
        
        # Colors for visualization (BGR format)
        self.FINGER_COLORS = {
            'INDEX': (0, 255, 0),    # Green
            'MIDDLE': (255, 0, 0),   # Blue
            'RING': (0, 0, 255),     # Red
            'PINKY': (255, 255, 0)   # Cyan
        }

    def process_frame(self, frame):
        # Flip the frame horizontally for a mirrored view
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        result = self.hands.process(frame_rgb)
        current_time = time.time()

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Get thumb tip coordinates
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                h, w, _ = frame.shape
                thumb_coords = np.array([int(thumb_tip.x * w), int(thumb_tip.y * h)])
                
                # Draw thumb point
                cv2.circle(frame, tuple(thumb_coords), 5, (255, 255, 255), -1)
                
                # Process each finger
                for finger_name, finger_landmark in self.FINGER_LANDMARKS.items():
                    finger_tip = hand_landmarks.landmark[finger_landmark]
                    finger_coords = np.array([int(finger_tip.x * w), int(finger_tip.y * h)])
                    
                    # Calculate distance between thumb and finger
                    distance = np.linalg.norm(finger_coords - thumb_coords)
                    
                    # Draw finger point and connection line
                    color = self.FINGER_COLORS[finger_name]
                    cv2.circle(frame, tuple(finger_coords), 5, color, -1)
                    cv2.line(frame, tuple(thumb_coords), tuple(finger_coords), color, 2)
                    
                    # Display distance
                    midpoint = ((thumb_coords + finger_coords) // 2)
                    cv2.putText(frame, f'{int(distance)}', tuple(midpoint), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Detect tap with cooldown
                    if (distance < self.DISTANCE_THRESHOLD and 
                        (current_time - self.last_tap_times[finger_name]) > self.COOLDOWN_PERIOD):
                        self.tap_counts[finger_name] += 1
                        self.last_tap_times[finger_name] = current_time
                        print(f"{finger_name} finger tap detected!")
                
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Display tap counts
        y_offset = 30
        for finger_name, count in self.tap_counts.items():
            cv2.putText(frame, f'{finger_name}: {count} taps', 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       self.FINGER_COLORS[finger_name], 2)
            y_offset += 30

        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame = self.process_frame(frame)
            cv2.imshow('Multi-Finger Tap Detection', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    detector = FingerTapDetector()
    detector.run()