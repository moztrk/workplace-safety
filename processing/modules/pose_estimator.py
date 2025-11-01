import cv2
import mediapipe as mp

class PoseEstimator:
    """MediaPipe Pose modelini yüklemek ve duruş tespiti yapmak için wrapper sınıf."""
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        print("Pose Estimator: Model yükleniyor...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        print("Pose Estimator: Model başarıyla yüklendi.")

    def estimate_pose(self, frame):
        """Verilen bir kare üzerinde duruş analizi yapar."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.pose.process(rgb_frame)
        rgb_frame.flags.writeable = True
        return results.pose_landmarks

    def draw_landmarks(self, frame, landmarks):
        """Tespit edilen iskeleti kare üzerine çizer."""
        if landmarks:
            self.mp_drawing.draw_landmarks(
                frame, 
                landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
        return frame
    
    def get_landmark_pixel(self, landmarks, landmark_id, frame_width, frame_height):
        """Belirli bir landmark'ın piksel koordinatlarını döndürür."""
        if landmarks:
            try:
                landmark = landmarks.landmark[landmark_id]
                if landmark.visibility < 0.5:
                    return None, None
                    
                pixel_x = int(landmark.x * frame_width)
                pixel_y = int(landmark.y * frame_height)
                return pixel_x, pixel_y
            except:
                return None, None
        return None, None