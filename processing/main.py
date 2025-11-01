import cv2
from guardian_processor import GuardianProcessor

# --- Global Ayarlar ---
YOLO_MODEL_PATH = "ppe.pt" 
KAYNAK = 0
WINDOW_NAME = "Guardian AI - (Refactored)" # <-- Pencere adını sabitle

# --- Adım 3: Tehlikeli Bölge Ayarları (FG-3) ---
# Kullanıcının tıkladığı noktaları saklamak için global liste
polygon_points = []

def mouse_callback(event, x, y, flags, param):
    """
    Fare tıklamalarını yöneten fonksiyon.
    """
    global polygon_points

    if event == cv2.EVENT_LBUTTONDOWN:
        # Sol tıklandığında: Listeye yeni bir köşe ekle
        polygon_points.append((x, y))
        print(f"Nokta eklendi: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Sağ tıklandığında: Poligonu sıfırla/temizle
        polygon_points.clear()
        print("Tehlikeli alan temizlendi.")


def main():
    try:
        processor = GuardianProcessor(YOLO_MODEL_PATH)
    except Exception as e:
        print(f"HATA: Guardian Processor başlatılamadı. Hata: {e}")
        return

    cap = cv2.VideoCapture(KAYNAK)
    if not cap.isOpened():
        print(f"Hata: Video kaynağı ({KAYNAK}) açılamadı.")
        return
        
    # PENCEREYİ DÖNGÜDEN ÖNCE OLUŞTURUYORUZ
    cv2.namedWindow(WINDOW_NAME)
    # FARE TIKLAMALARINI BU PENCEREYE BAĞLIYORUZ
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("Kamera açıldı. Çıkış için 'q' tuşuna basın.")
    print("--- Tehlikeli Alan Çizimi ---")
    print("  Sol Tık: Alana köşe ekle")
    print("  Sağ Tık: Alanı temizle")
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Global 'polygon_points' listesini işlemciye gönder
        annotated_frame, raw_data = processor.process_frame(frame, polygon_points)
        
        # 'raw_data' içindeki tehlike durumunu al
        is_in_danger = raw_data.get("is_inside_danger_zone", False)
        
        # Ekrana durumu yazdır (Opsiyonel ama faydalı)
        status_text = "TEHLIKEDE!" if is_in_danger else "GUVENDE"
        color = (0, 0, 255) if is_in_danger else (0, 255, 0)
        cv2.putText(annotated_frame, status_text, (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Sonucu her zaman aynı isimli pencerede göster
        cv2.imshow(WINDOW_NAME, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    print("Akış sonlandırılıyor...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()