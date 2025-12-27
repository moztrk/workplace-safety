import cv2
from ultralytics import YOLO

class YoloDetector:
    """YOLOv8 modelini yüklemek ve nesne tespiti yapmak için wrapper sınıf."""
    
    def __init__(self, model_path):
        print(f"YOLO Detector: Model yükleniyor... ({model_path})")
        try:
            self.model = YOLO(model_path)
            print("YOLO Detector: Model başarıyla yüklendi.")
        except Exception as e:
            print(f"HATA: YOLO modeli yüklenemedi! Hata: {e}")
            raise e

    def detect_objects(self, frame):
        """YOLO modelini kullanarak nesneleri algılar."""
        results = self.model(
            frame, 
            conf=0.4,      # Minimum güven skoru (0.3'ten artırıldı)
            iou=0.3,       # Daha agresif NMS (overlap azaltıldı)
            max_det=30     # Maksimum deteksiyon sayısı
        )
        return results[0]

    def draw_detections(self, frame, results):
        """Tespit sonuçlarını kare üzerine çizer."""
        annotated_frame = results.plot()
        return annotated_frame