import cv2
from tracker.tracker import Tracker
from ultralytics import YOLO

def main():
    model_path = 'model/yolov8n-face.pt'
    tracker = Tracker(model_path)
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = tracker.detect_frames(frame)
        tracked_objects = tracker.get_object_tracks(frame, detections)
        
        for obj in tracked_objects:
            x1, y1, x2, y2, score = obj
            tracker.create_rectangle(frame, x1, y1, x2, y2)
            tracker.create_landmarks(frame, x1, y1, x2, y2)
        
        cv2.imshow("Real-Time Face Detection and Landmark Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
