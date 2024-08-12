import cv2
import cvzone
import supervision as sv
from ultralytics import YOLO
import mediapipe as mp

class Tracker:
    def __init__(self, model_path):
        # Load the YOLO model
        self.model = YOLO(model_path)
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, min_detection_confidence=0.4)
        
        # Landmark indices for eyes, nose, and mouth
        self.left_eye_indices = [33, 133]
        self.right_eye_indices = [362, 263]
        self.nose_indices = [1, 4, 6, 195, 5]
        self.mouth_indices = [61, 81, 13, 14, 17]
    
    def detect_frames(self, frames):
        """Detect faces in frames and draw bounding boxes."""
        face_result = self.model.track(frames, conf=0.4)
        detections = []
        for info in face_result:
            parameters = info.boxes
            for box in parameters:
                score = box.data[0, 4].item()
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                h, w = y2 - y1, x2 - x1
                cvzone.cornerRect(frames, [x1, y1, w, h], l=9, rt=3)
                detections.append([x1, x2, y1, y2, score])
        return detections
    
    def get_object_tracks(self, frames, detections):
        """Get object tracks based on detections."""
        tracked_objects = []
        for detection in detections:
            x1, x2, y1, y2, score = detection
            tracked_objects.append((x1, y1, x2, y2, score))
        return tracked_objects
    
    def create_landmarks(self, frames, x1, y1, x2, y2):
        """Draw landmarks on detected faces."""
        # Extract face ROI
        face_roi = frames[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
        # Detect facial landmarks with MediaPipe
        results = self.face_mesh.process(face_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw landmarks for left eye
                for idx in self.left_eye_indices:
                    x = int(face_landmarks.landmark[idx].x * (x2 - x1)) + x1
                    y = int(face_landmarks.landmark[idx].y * (y2 - y1)) + y1
                    cv2.circle(frames, (x, y), 1, (0, 255, 0), -1)

                # Draw landmarks for right eye
                for idx in self.right_eye_indices:
                    x = int(face_landmarks.landmark[idx].x * (x2 - x1)) + x1
                    y = int(face_landmarks.landmark[idx].y * (y2 - y1)) + y1
                    cv2.circle(frames, (x, y), 1, (0, 255, 0), -1)
                
                # Draw landmarks for nose
                for idx in self.nose_indices:
                    x = int(face_landmarks.landmark[idx].x * (x2 - x1)) + x1
                    y = int(face_landmarks.landmark[idx].y * (y2 - y1)) + y1
                    cv2.circle(frames, (x, y), 1, (0, 255, 0), -1)

                # Draw landmarks for mouth
                for idx in self.mouth_indices:
                    x = int(face_landmarks.landmark[idx].x * (x2 - x1)) + x1
                    y = int(face_landmarks.landmark[idx].y * (y2 - y1)) + y1
                    cv2.circle(frames, (x, y), 1, (0, 255, 0), -1)
    
    def create_rectangle(self, frames, x1, y1, x2, y2):
        """Draw a rectangle around the detected face."""
        cv2.rectangle(frames, (x1, y1), (x2, y2), (255, 0, 0), 2)
