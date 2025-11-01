import cv2
import mediapipe as mp

class PoseEstimator:
    """
    MediaPipe Pose modelini yüklemek ve duruş tespiti yapmak için
    kapsülleyici (wrapper) sınıf.
    """
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Modeli başlatır ve yükler.
        """
        print("Pose Estimator: Model yükleniyor...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        print("Pose Estimator: Model başarıyla yüklendi.")

    def estimate_pose(self, frame):
        """
        Verilen bir kare (frame) üzerinde duruş analizi yapar.
        
        Args:
            frame: Üzerinde analiz yapılacak BGR video karesi.
            
        Returns:
            MediaPipe 'pose_landmarks' nesnesi (veya bulunamazsa None).
        """
        # MediaPipe RGB formatında görüntü bekler, OpenCV BGR verir.
        # Renge dönüştür.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Performans için görüntüyü "yazılamaz" olarak işaretle
        rgb_frame.flags.writeable = False
        
        # Duruş analizini yap
        results = self.pose.process(rgb_frame)
        
        # Görüntüyü tekrar "yazılabilir" yap
        rgb_frame.flags.writeable = True
        
        # Sadece ham landmark (eklem) verisini döndür
        return results.pose_landmarks

    def draw_landmarks(self, frame, landmarks):
        """
        Tespit edilen iskeleti (landmarks) verilen kare üzerine çizer.
        
        Args:
            frame: Çizim yapılacak orijinal BGR kare.
            landmarks: estimate_pose() fonksiyonundan dönen landmark verisi.
            
        Returns:
            Üzerine çizim yapılmış kare (annotated_frame).
        """
        # Eğer bir iskelet bulunduysa, çiz
        if landmarks:
            self.mp_drawing.draw_landmarks(
                frame, 
                landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
        return frame
    def get_landmark_pixel(self, landmarks, landmark_id, frame_width, frame_height):
        """
        Belirli bir landmark'ın ID'sini alır ve onun piksel 
        koordinatlarını (x, y) olarak döndürür.
        
        Args:
            landmarks: MediaPipe'tan dönen 'pose_landmarks' nesnesi.
            landmark_id: İstenen noktanın ID'si (örn: mp_pose.PoseLandmark.LEFT_ANKLE).
            frame_width (int): Video karesinin genişliği.
            frame_height (int): Video karesinin yüksekliği.
            
        Returns:
            tuple: (x, y) piksel koordinatları veya (None, None).
        """
        if landmarks:
            try:
                # İstenen landmark'ın normalize koordinatlarını al
                landmark = landmarks.landmark[landmark_id]
                
                # Görünürlüğünü kontrol et (ekran dışında veya kapalı değilse)
                if landmark.visibility < 0.5:
                    return None, None
                    
                # Piksel koordinatlarına çevir
                pixel_x = int(landmark.x * frame_width)
                pixel_y = int(landmark.y * frame_height)
                
                return pixel_x, pixel_y
            except:
                return None, None
        return None, None