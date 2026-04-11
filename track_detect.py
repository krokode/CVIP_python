import cv2
from ultralytics import YOLO

class SoccerTracker:
    def __init__(self, model_path='yolov8s.pt'):
        self.model = YOLO(model_path) 
        self.tracker = None
        self.is_tracking = False
        self.bbox = None
        self.status = "Initializing"
        self.frame_count = 0 
        self.redetect_interval = 30 # Force YOLO every 30 frames to prevent drift

    def get_tracker(self, tracker_type='CSRT'):
        # Try to get the better trackers first, fallback to MIL
        if tracker_type == 'CSRT' and hasattr(cv2, 'TrackerCSRT_create'):
            return cv2.TrackerCSRT_create()
        if tracker_type == 'KCF' and hasattr(cv2, 'TrackerKCF_create'):
            return cv2.TrackerKCF_create()
        return cv2.TrackerMIL_create()

    def detect_and_track(self, frame, class_id=32, tracker_name='CSRT'):
        self.frame_count += 1
        
        if self.frame_count % self.redetect_interval == 0:
            self.is_tracking = False

        if not self.is_tracking:
            # DETECTION STATE (Blue)
            results = self.model.predict(frame, conf=0.4, classes=[class_id], verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            if len(boxes) > 0:
                x1, y1, x2, y2 = boxes[0]
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                
                # Clamp bbox to frame boundaries and ensure positive dimensions
                h_frame, w_frame = frame.shape[:2]
                x = max(0, min(x, w_frame - 1))
                y = max(0, min(y, h_frame - 1))
                w = max(20, min(w, w_frame - x))
                h = max(20, min(h, h_frame - y))
                
                self.bbox = (x, y, w, h)
                
                self.tracker = self.get_tracker(tracker_type=tracker_name)
                self.tracker.init(frame, self.bbox) 
                self.is_tracking = True
                self.status = "Detection (Blue)"
                return self.bbox, (255, 0, 0) # BLUE
            
            self.status = "Searching..."
            return None, (0, 0, 255)

        else:
            # TRACKING STATE (Green)
            success, new_bbox = self.tracker.update(frame)
            
            # Check if the tracker went out of bounds or failed
            if success:
                self.bbox = new_bbox
                self.status = "Tracking (Green)"
                return self.bbox, (0, 255, 0) # GREEN
            else:
                self.is_tracking = False 
                self.status = "Lost - Redetecting"
                return None, (0, 0, 255)

if __name__ == "__main__":
    
    cap = cv2.VideoCapture("soccer-ball.mp4")
    soccer_tracker = SoccerTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        bbox, color = soccer_tracker.detect_and_track(frame, class_id=32)
        
        if bbox is not None:
            x, y, w, h = [int(v) for v in bbox]
            # Draw box and status
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, soccer_tracker.status, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow("Soccer Detection & Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
