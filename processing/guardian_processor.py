import cv2
import numpy as np  # <-- YENİ EKLENDİ
from modules.object_detector import YoloDetector
from modules.pose_estimator import PoseEstimator

class GuardianProcessor:
    """
    Tüm analiz modüllerini (YOLO, MediaPipe) yöneten,
    orkestra şefi sınıfı.
    """
    
    def __init__(self, yolo_model_path):
        """
        Gerekli tüm alt modülleri başlatır.
        
        Args:
            yolo_model_path (str): YOLO .pt model dosyasının yolu.
        """
        self.yolo_detector = YoloDetector(yolo_model_path)
        self.pose_estimator = PoseEstimator()
        
        # Kontrol için sol ayak bileği ID'sini sakla
        self.check_landmark_id = self.pose_estimator.mp_pose.PoseLandmark.LEFT_ANKLE

    def process_frame(self, frame):
        """
        Tek bir video karesini alır ve tüm analiz adımlarını uygular.
        
        Args:
            frame: İşlenecek video karesi.
            
        Returns:
            tuple: (annotated_frame, raw_data)
                - annotated_frame: Üzerine çizim yapılmış video karesi.
                - raw_data: Gelecek adımlarda analiz için kullanılacak ham veri.
        """
        
        # --- Adım 1: Nesne Tespiti (YOLO) ---
        # Ham YOLO sonuçlarını al (kutucuklar, sınıflar, güven skorları)
        yolo_results = self.yolo_detector.detect_objects(frame)
        
        # --- Adım 2: Duruş Analizi (MediaPipe) ---
        # Ham iskelet (landmark) verisini al
        pose_landmarks = self.pose_estimator.estimate_pose(frame)
        
        # --- Adım 3: Görselleştirme ---
        # Çizime orijinal kareden başla
        annotated_frame = frame.copy()
        
        # Önce YOLO kutucuklarını çiz
        annotated_frame = self.yolo_detector.draw_detections(annotated_frame, yolo_results)
        
        # Sonra iskeleti üzerine çiz
        annotated_frame = self.pose_estimator.draw_landmarks(annotated_frame, pose_landmarks)

        # --- Adım 4: Veri Toplama ---
        # Gelecekteki "Akıllı Uyarı Sistemi" (Adım 4) için
        # ham verileri bir sözlükte topla.
    def process_frame(self, frame, polygon_points):
        """
        Tek bir video karesini alır ve tüm analiz adımlarını uygular.
        
        Args:
            frame: İşlenecek video karesi.
            polygon_points (list): Kullanıcının tıkladığı poligon köşeleri.
            
        Returns:
            tuple: (annotated_frame, raw_data)
        """
        
        # --- Ham Veri Toplama ---
        frame_height, frame_width, _ = frame.shape
        yolo_results = self.yolo_detector.detect_objects(frame)
        pose_landmarks = self.pose_estimator.estimate_pose(frame)
        
        # --- Adım 3: Tehlikeli Bölge Kontrolü (FG-4) ---
        is_inside = False  # Varsayılan olarak tehlikede değil
        
        # 1. Yeterli poligon noktası (en az 3) ve bir iskelet var mı?
        if len(polygon_points) >= 3 and pose_landmarks:
            
            # 2. Kontrol edeceğimiz noktanın (ayak bileği) piksel koordinatını al
            check_px, check_py = self.pose_estimator.get_landmark_pixel(
                pose_landmarks, 
                self.check_landmark_id, 
                frame_width, 
                frame_height
            )
            
            # 3. Eğer nokta görünürse (None değilse)
            if check_px is not None:
                foot_point = (check_px, check_py)
                
                # 4. Poligonu OpenCV'nin anlayacağı formata (Numpy array) çevir
                polygon_np = np.array(polygon_points, np.int32)
                
                # 5. EFSANE FONKSİYON: Nokta, poligonun içinde mi?
                #   > 0: İçinde, == 0: Kenarında, < 0: Dışında
                distance = cv2.pointPolygonTest(polygon_np, foot_point, False)
                
                if distance >= 0:
                    is_inside = True # Tehlikede!
                    
        # --- Adım 4: Görselleştirme ---
        annotated_frame = frame.copy()
        
        # 1. YOLO kutucuklarını çiz
        annotated_frame = self.yolo_detector.draw_detections(annotated_frame, yolo_results)
        
        # 2. MediaPipe iskeletini çiz
        annotated_frame = self.pose_estimator.draw_landmarks(annotated_frame, pose_landmarks)
        
        # 3. Tehlikeli Alan Poligonunu Çiz (FG-3)
        if len(polygon_points) > 0:
            # Poligonun rengini duruma göre belirle
            poly_color_bgr = (0, 0, 255) if is_inside else (0, 255, 0)
            
            # Kullanıcının tıkladığı noktaları (köşeleri) çiz
            for point in polygon_points:
                cv2.circle(annotated_frame, point, 5, (255, 0, 0), -1) # Mavi noktalar
            
            # Poligonun çizgilerini çiz
            if len(polygon_points) >= 2:
                polygon_np = np.array([polygon_points], np.int32)
                cv2.polylines(annotated_frame, polygon_np, isClosed=True, color=poly_color_bgr, thickness=2)

        # --- Veri Toplama ---
        raw_data = {
            "yolo_results": yolo_results,
            "pose_landmarks": pose_landmarks,
            "is_inside_danger_zone": is_inside # <-- Adım 4 için önemli veri
        }
        
        return annotated_frame, raw_data