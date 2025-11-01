import cv2
import numpy as np
from modules.object_detector import YoloDetector
from modules.pose_estimator import PoseEstimator

class GuardianProcessor:
    """Tüm analiz modüllerini (YOLO, MediaPipe) yöneten orkestra şefi sınıfı."""
    
    def __init__(self, yolo_model_path):
        self.yolo_detector = YoloDetector(yolo_model_path)
        self.pose_estimator = PoseEstimator()
        self.check_landmark_id = self.pose_estimator.mp_pose.PoseLandmark.LEFT_ANKLE

    def process_frame(self, frame, polygon_points):
        """Tek bir video karesini alır ve tüm analiz adımlarını uygular."""
        
        # --- Ham Veri Toplama ---
        frame_height, frame_width, _ = frame.shape
        yolo_results = self.yolo_detector.detect_objects(frame)
        pose_landmarks = self.pose_estimator.estimate_pose(frame)
        
        # --- Tehlikeli Bölge Kontrolü ---
        is_inside = False
        
        if len(polygon_points) >= 3 and pose_landmarks:
            check_px, check_py = self.pose_estimator.get_landmark_pixel(
                pose_landmarks, 
                self.check_landmark_id, 
                frame_width, 
                frame_height
            )
            
            if check_px is not None:
                foot_point = (check_px, check_py)
                polygon_np = np.array(polygon_points, np.int32)
                distance = cv2.pointPolygonTest(polygon_np, foot_point, False)
                
                if distance >= 0:
                    is_inside = True
                    
        # --- Görselleştirme ---
        annotated_frame = frame.copy()
        annotated_frame = self.yolo_detector.draw_detections(annotated_frame, yolo_results)
        annotated_frame = self.pose_estimator.draw_landmarks(annotated_frame, pose_landmarks)
        
        # Tehlikeli Alan Poligonunu Çiz
        if len(polygon_points) > 0:
            poly_color_bgr = (0, 0, 255) if is_inside else (0, 255, 0)
            
            for point in polygon_points:
                cv2.circle(annotated_frame, point, 5, (255, 0, 0), -1)
            
            if len(polygon_points) >= 2:
                polygon_np = np.array([polygon_points], np.int32)
                cv2.polylines(annotated_frame, polygon_np, isClosed=True, 
                             color=poly_color_bgr, thickness=2)

        # --- Veri Toplama ---
        raw_data = {
            "yolo_results": yolo_results,
            "pose_landmarks": pose_landmarks,
            "is_inside_danger_zone": is_inside
        }
        
        return annotated_frame, raw_data