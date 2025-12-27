import cv2
import json
import os
from guardian_processor import GuardianProcessor

# --- Global Ayarlar ---
YOLO_MODEL_PATH = "best.pt"  # processing klasöründe olduğu için sadece dosya adı yeterli
# Video dosyası kullanmak için dosya adını buraya yazın
KAYNAK = "test1.mp4"  # Kamera için: 0
WINDOW_NAME = "Guardian AI - Workplace Safety"
POLYGON_FILE = "danger_zone.json"  # Poligon kayıt dosyası

# --- Global Değişkenler ---
polygon_points = []
is_locked = False  # Kilit durumu

def save_polygon(points, filename=POLYGON_FILE):
    """Poligon noktalarını JSON dosyasına kaydeder."""
    if len(points) >= 1:  # En az 1 nokta varsa kaydet
        with open(filename, 'w') as f:
            json.dump(points, f)
        print(f"✓ Tehlikeli alan kaydedildi: {filename} ({len(points)} nokta)")
    else:
        print("⚠️  Kaydedilecek nokta yok!")

def load_polygon(filename=POLYGON_FILE):
    """JSON dosyasından poligon noktalarını yükler."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            points = json.load(f)
        print(f"Tehlikeli alan yüklendi: {len(points)} nokta")
        return points
    return []

def delete_polygon(filename=POLYGON_FILE):
    """JSON dosyasındaki kaydedilmiş poligonu siler."""
    if os.path.exists(filename):
        os.remove(filename)
        print(f"✓ Kaydedilmiş tehlikeli alan silindi: {filename}")
        return True
    else:
        print("⚠️  Silinecek kayıtlı alan bulunamadı.")
        return False

def mouse_callback(event, x, y, flags, param):
    """Fare tıklamalarını yöneten fonksiyon (Sadece çizim modunda aktif)."""
    global polygon_points, is_locked
    
    # Eğer kilitliyse tıklamaları engelle
    if is_locked:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        print(f"✓ Nokta eklendi: ({x}, {y}) - Toplam: {len(polygon_points)}")
    elif event == cv2.EVENT_RBUTTONDOWN:
        polygon_points.clear()
        print("🗑️ Tüm noktalar silindi.")
    elif event == cv2.EVENT_MBUTTONDOWN:
        if len(polygon_points) > 0:
            removed = polygon_points.pop()
            print(f"⬅️ Son nokta silindi: {removed}")

def main():
    global polygon_points, is_locked
    
    try:
        processor = GuardianProcessor(YOLO_MODEL_PATH)
    except Exception as e:
        print(f"❌ HATA: Guardian Processor başlatılamadı. {e}")
        return

    cap = cv2.VideoCapture(KAYNAK)
    if not cap.isOpened():
        print(f"❌ Video kaynağı açılamadı: {KAYNAK}")
        return
        
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    # Kaydedilmiş poligonu yükle (ama kilitleme)
    if os.path.exists(POLYGON_FILE):
        with open(POLYGON_FILE, 'r') as f:
            polygon_points = json.load(f)
        print(f"📂 Alan yüklendi ({len(polygon_points)} nokta). Takip için 'L' tuşuna basın.")

    print("\n" + "="*60)
    print("🎯 GUARDIAN AI - İŞ GÜVENLİĞİ SİSTEMİ")
    print("="*60)
    print("\n⌨️  KONTROLLER:")
    print("  'L' - TAKİBİ BAŞLAT (Çizimi kilitle, kamera hareketi takibi aktif)")
    print("  'R' - TAKİBİ DURDUR (Kilidi aç, yeniden çizim yapabilirsiniz)")
    print("  'M' - GÖRÜNÜM MODU (minimal → normal → full)")
    print("  'S' - Poligonu kaydet (danger_zone.json)")
    print("  'C' - Kaydedilmiş poligonu sil")
    print("  'Q' veya ESC - Çıkış")
    print("\n🖱️  FARE:")
    print("  Sol Tık   - Nokta ekle (çizim modunda)")
    print("  Sağ Tık   - Tüm noktaları sil")
    print("  Orta Tık  - Son noktayı sil")
    print("="*60 + "\n")
    
    frame_count = 0
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                if isinstance(KAYNAK, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            
            frame_count += 1
            frame = cv2.resize(frame, (854, 480))
            
            # Process frame
            annotated_frame, data = processor.process_frame(frame, polygon_points)
            
            # Frame sayacı (Sadece full modda)
            if processor.display_mode == "full":
                cv2.putText(annotated_frame, f"Frame: {frame_count}", 
                           (annotated_frame.shape[1] - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Kullanıcı ipucu (çizim modundayken ve minimal değilse)
            if not is_locked and processor.display_mode != "minimal":
                hint_text = "Cizim yapin ve 'L' tusuna basin (Kilitle)"
                cv2.putText(annotated_frame, hint_text, (20, annotated_frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow(WINDOW_NAME, annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q veya ESC
                break
                
            elif key == ord('l') or key == ord('L'):  # LOCK
                if len(polygon_points) >= 3:
                    success = processor.start_tracking(frame, polygon_points)
                    if success:
                        is_locked = True
                        print("🔒 Takip başlatıldı!")
                else:
                    print("⚠️ En az 3 nokta gerekli!")
                    
            elif key == ord('r') or key == ord('R'):  # RESET
                processor.stop_tracking()
                is_locked = False
                polygon_points.clear()
                print("🔓 Takip durduruldu.")
                
            elif key == ord('m') or key == ord('M'):  # MODE SWITCH
                modes = ["minimal", "normal", "full"]
                current_idx = modes.index(processor.display_mode)
                next_mode = modes[(current_idx + 1) % len(modes)]
                processor.set_display_mode(next_mode)
                
            elif key == ord('s') or key == ord('S'):  # SAVE
                if len(polygon_points) >= 3:
                    with open(POLYGON_FILE, 'w') as f:
                        json.dump(polygon_points, f)
                    print(f"💾 Alan kaydedildi ({len(polygon_points)} nokta).")
                else:
                    print("⚠️ En az 3 nokta kaydetmelisiniz!")
                    
            elif key == ord('c') or key == ord('C'):  # CLEAR
                if os.path.exists(POLYGON_FILE):
                    os.remove(POLYGON_FILE)
                    print("🗑️ Kaydedilmiş alan silindi.")
                processor.stop_tracking()
                is_locked = False
                polygon_points.clear()
            
            # Pencere kapatma kontrolü
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
                
    except KeyboardInterrupt:
        print("\n⚠️ Kullanıcı tarafından durduruldu (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Hata: {e}")
    finally:
        print("\n✓ Program kapatılıyor...")
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print("✓ Kaynaklar temizlendi.")

if __name__ == "__main__":
    main()