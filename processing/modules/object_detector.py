from ultralytics import YOLO

class YoloDetector:
    """
    YOLOv8 modelini yüklemek ve nesne tespiti yapmak için 
    kapsülleyici (wrapper) sınıf.
    """
    
    def __init__(self, model_path):
        """
        Modeli başlatır ve yükler.
        
        Args:
            model_path (str): Kullanılacak .pt model dosyasının yolu.
        """
        print(f"YOLO Detector: Model yükleniyor... ({model_path})")
        try:
            self.model = YOLO(model_path)
            print("YOLO Detector: Model başarıyla yüklendi.")
        except Exception as e:
            print(f"HATA: YOLO modeli yüklenemedi! Hata: {e}")
            # Hata durumunda programın çökmesi yerine None atayabiliriz,
            # ancak bu senaryoda modelin yüklenmesi kritik olduğu için 
            # hatayı yükseltmek daha doğru olabilir.
            raise e

    def detect_objects(self, frame):
        """
        Verilen bir kare (frame) üzerinde nesne tespiti yapar.
        
        Args:
            frame: Üzerinde tespit yapılacak video karesi (numpy array).
            
        Returns:
            Ultralytics 'Results' nesnesi.
        """
        # Modeli frame üzerinde çalıştır
        results = self.model(frame)
        
        # Sadece ilk sonucu (tek bir görüntü işlediğimiz için) döndür
        return results[0]

    def draw_detections(self, frame, results):
        """
        Tespit sonuçlarını (kutucuklar, etiketler) verilen kare üzerine çizer.
        
        Args:
            frame: Çizim yapılacak orijinal kare.
            results: detect_objects() fonksiyonundan dönen 'Results' nesnesi.
            
        Returns:
            Üzerine çizim yapılmış kare (annotated_frame).
        """
        # Ultralytics'in kendi 'plot' fonksiyonunu kullan
        annotated_frame = results.plot()
        return annotated_frame