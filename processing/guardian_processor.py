import cv2
import numpy as np
import os
from datetime import datetime
from modules.object_detector import YoloDetector
from modules.pose_estimator import PoseEstimator

class GuardianProcessor:
    """Tüm analiz modüllerini (YOLO, MediaPipe) yöneten orkestra şefi sınıfı."""
    
    def __init__(self, yolo_model_path):
        self.yolo_detector = YoloDetector(yolo_model_path)
        self.pose_estimator = PoseEstimator()
        self.check_landmark_id = self.pose_estimator.mp_pose.PoseLandmark.LEFT_ANKLE
        self.yolo_class_names = self.yolo_detector.model.names
        
        # Fotoğraf kaydı için sayaç/zamanlayıcı
        self.last_save_time = 0
        if not os.path.exists("violations"):
            os.makedirs("violations")
        
        # --- Takip Değişkenleri (Güçlendirilmiş) ---
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.reference_frame = None
        self.reference_keypoints = None
        self.reference_descriptors = None
        self.original_polygon = None
        self.tracking_enabled = False
        
        # --- Görünüm Modu ---
        self.display_mode = "minimal"  # "minimal", "normal", "full"

    def set_display_mode(self, mode):
        """Görünüm modunu değiştirir."""
        if mode in ["minimal", "normal", "full"]:
            self.display_mode = mode
            print(f"📺 Görünüm modu: {mode.upper()}")

    def _calculate_iou(self, box1, box2):
        """İki kutunun kesişim oranını hesaplar."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Kesişim alanı
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        inter_width = max(0, inter_xmax - inter_xmin)
        inter_height = max(0, inter_ymax - inter_ymin)
        inter_area = inter_width * inter_height
        
        # Birleşim alanı
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0
        
        return inter_area / union_area

    def _match_ppe_to_person(self, yolo_results, frame_shape):
        """Her Person için yakınındaki KKD'leri eşleştirir (Bölge tabanlı)."""
        persons = []
        ppe_items = []
        
        # Deteksiyonları ayır
        for box, cls_id, conf in zip(yolo_results.boxes.xyxy, 
                                      yolo_results.boxes.cls, 
                                      yolo_results.boxes.conf):
            class_name = self.yolo_class_names[int(cls_id)]
            x1, y1, x2, y2 = map(int, box)
            
            if class_name == "Person":
                persons.append({
                    'box': (x1, y1, x2, y2),
                    'center': ((x1+x2)//2, (y1+y2)//2),
                    'foot': ((x1+x2)//2, y2),
                    'has_helmet': False,
                    'has_vest': False,
                    'helmet_conf': 0.0,
                    'vest_conf': 0.0
                })
            elif class_name in ["Hardhat", "Safety Vest"]:
                ppe_items.append({
                    'type': class_name,
                    'box': (x1, y1, x2, y2),
                    'center': ((x1+x2)//2, (y1+y2)//2),
                    'conf': float(conf),
                    'assigned': False  # KKD zaten atandı mı?
                })
        
        # Her KKD için en yakın Person'ı bul (çakışmayı önlemek için)
        for ppe in ppe_items:
            if ppe['assigned']:
                continue
                
            ppx, ppy = ppe['center']
            min_distance = float('inf')
            closest_person = None
            
            for person in persons:
                px1, py1, px2, py2 = person['box']
                person_width = px2 - px1
                person_height = py2 - py1
                
                # Genişletilmiş arama alanı
                search_area = (
                    px1 - person_width * 1.2,
                    py1 - person_height * 0.7,
                    px2 + person_width * 1.2,
                    py2 + person_height * 0.3
                )
                
                # KKD bu alanda mı?
                if (search_area[0] <= ppx <= search_area[2] and 
                    search_area[1] <= ppy <= search_area[3]):
                    
                    # Mesafeyi hesapla
                    person_center_x = (px1 + px2) / 2
                    person_center_y = (py1 + py2) / 2
                    distance = ((ppx - person_center_x)**2 + (ppy - person_center_y)**2)**0.5
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_person = person
            
            # En yakın Person'a KKD'yi ata
            if closest_person:
                if ppe['type'] == "Hardhat":
                    person_mid_y = (closest_person['box'][1] + closest_person['box'][3]) / 2
                    if ppy < person_mid_y:
                        # Daha yüksek confidence varsa değiştir
                        if ppe['conf'] > closest_person['helmet_conf']:
                            closest_person['has_helmet'] = True
                            closest_person['helmet_conf'] = ppe['conf']
                            ppe['assigned'] = True
                        
                elif ppe['type'] == "Safety Vest":
                    # Daha yüksek confidence varsa değiştir
                    if ppe['conf'] > closest_person['vest_conf']:
                        closest_person['has_vest'] = True
                        closest_person['vest_conf'] = ppe['conf']
                        ppe['assigned'] = True
        
        return persons

    def _analyze_ppe_status(self, yolo_results):
        """YOLO sonuçlarından KKD eksikliğini tespit eder."""
        has_no_helmet = any(self.yolo_class_names[int(cls_id)] == 'NO-Hardhat' 
                           for cls_id in yolo_results.boxes.cls)
        has_no_vest = any(self.yolo_class_names[int(cls_id)] == 'NO-Safety Vest' 
                         for cls_id in yolo_results.boxes.cls)
        return has_no_helmet, has_no_vest

    def _run_rule_engine(self, ppe_status, is_inside_zone):
        """Risk seviyesini ve uyarı mesajını belirler."""
        has_no_helmet, has_no_vest = ppe_status
        risk_level, alert_message, color_bgr = "GUVENDE", "GUVENDE", (0, 255, 0)

        if (has_no_helmet or has_no_vest) and is_inside_zone:
            risk_level = "KRITIK"
            alert_message = "KRITIK IHLAL: ALAN + KKD EKSIK!"
            color_bgr = (0, 0, 255)
        elif has_no_helmet or has_no_vest:
            risk_level = "ORTA"
            alert_message = "UYARI: KKD EKSIK"
            color_bgr = (0, 165, 255)
        elif is_inside_zone:
            risk_level = "DUSUK"
            alert_message = "DIKKAT: TEHLIKELI BOLGE"
            color_bgr = (0, 255, 255)
        
        return risk_level, alert_message, color_bgr

    def start_tracking(self, frame, polygon_points):
        """Çizim bittiğinde main.py tarafından manuel çağrılacak."""
        if len(polygon_points) < 3:
            print("⚠️ Takip için en az 3 nokta gerekli!")
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.reference_keypoints, self.reference_descriptors = self.orb.detectAndCompute(gray, None)
        
        if self.reference_descriptors is not None and len(self.reference_keypoints) > 20:
            self.reference_frame = gray.copy()
            self.original_polygon = list(polygon_points)  # Kopyasını al
            self.tracking_enabled = True
            print(f"✅ Referans alındı ve takip kilitlendi. ({len(self.reference_keypoints)} özellik noktası)")
            return True
        else:
            print("❌ Yetersiz görüntü özelliği, takip başlatılamadı.")
            return False

    def stop_tracking(self):
        """Takibi durdurur ve çizim moduna döner."""
        self.tracking_enabled = False
        self.reference_frame = None
        self.reference_descriptors = None
        self.original_polygon = None
        print("🛑 Takip durduruldu. Çizim moduna geçildi.")

    def _update_polygon_tracking(self, frame):
        """Kamera hareketine göre poligonu kaydırır (Homography)."""
        if not self.tracking_enabled or self.reference_descriptors is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_keypoints, current_descriptors = self.orb.detectAndCompute(gray, None)
        
        if current_descriptors is None or len(current_keypoints) < 10:
            return self.original_polygon
        
        try:
            matches = self.bf_matcher.match(self.reference_descriptors, current_descriptors)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # En iyi %30 eşleşmeyi al (Gürültüyü azaltır)
            keep_percent = 0.3
            keep_count = int(len(matches) * keep_percent)
            good_matches = matches[:max(keep_count, 10)]
            
            if len(good_matches) < 10:
                return self.original_polygon
            
            src_pts = np.float32([self.reference_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([current_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is not None:
                pts = np.float32(self.original_polygon).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                return [(int(x), int(y)) for x, y in dst.reshape(-1, 2)]
                
        except Exception as e:
            print(f"⚠️ Takip hatası: {e}")
        
        return self.original_polygon

    def process_frame(self, frame, current_draw_points):
        """Tek bir video karesini alır ve tüm analiz adımlarını uygular."""
        
        # 1. Adım: Hangi poligonu kullanacağız?
        if self.tracking_enabled:
            active_polygon = self._update_polygon_tracking(frame)
        else:
            active_polygon = current_draw_points
        
        # --- Ham Veri Toplama ---
        frame_height, frame_width, _ = frame.shape
        yolo_results = self.yolo_detector.detect_objects(frame)
        
        # Kişi-KKD eşleştirmesi yap (TÜM kişiler için)
        persons = self._match_ppe_to_person(yolo_results, frame.shape)
        
        # Tehlikeli bölgedeki kişileri bul
        persons_in_danger = []
        if len(active_polygon) >= 3:
            polygon_np = np.array(active_polygon, np.int32)
            
            for person in persons:
                foot_x, foot_y = person['foot']
                result = cv2.pointPolygonTest(polygon_np, (foot_x, foot_y), False)
                if result >= 0:
                    persons_in_danger.append(person)
        
        # Risk değerlendirmesi (sadece bölgedeki kişiler için)
        risk_level = "GUVENDE"
        alert_msg = "GUVENDE"
        alert_color = (0, 255, 0)
        
        if len(persons_in_danger) > 0:
            missing_ppe = []
            for person in persons_in_danger:
                if not person['has_helmet']:
                    missing_ppe.append("Baret")
                if not person['has_vest']:
                    missing_ppe.append("Yelek")
            
            if missing_ppe:
                risk_level = "KRITIK"
                alert_msg = f"KRITIK IHLAL: {', '.join(set(missing_ppe))} EKSIK!"
                alert_color = (0, 0, 255)
            else:
                risk_level = "DUSUK"
                alert_msg = "DIKKAT: TEHLIKELI BOLGEYE GIRIS (KKD Tamam)"
                alert_color = (0, 255, 255)
                    
        # --- Görselleştirme (Mod'a göre) ---
        annotated_frame = frame.copy()
        
        # 1. YOLO Deteksiyonları (Sadece normal/full modda)
        if self.display_mode in ["normal", "full"]:
            annotated_frame = self.yolo_detector.draw_detections(annotated_frame, yolo_results)
        
        # 2. Tehlikeli bölgedeki kişileri vurgula
        for person in persons_in_danger:
            x1, y1, x2, y2 = person['box']
            
            # Kalın çerçeve
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), alert_color, 4)
            
            # KKD durumu
            if person['has_helmet']:
                helmet_status = f"OK ({person['helmet_conf']:.2f})"
                helmet_color = (0, 255, 0)
            else:
                helmet_status = "YOK"
                helmet_color = (0, 0, 255)
            
            if person['has_vest']:
                vest_status = f"OK ({person['vest_conf']:.2f})"
                vest_color = (0, 255, 0)
            else:
                vest_status = "YOK"
                vest_color = (0, 0, 255)
            
            # Minimal modda sadece eksik olanları göster
            if self.display_mode == "minimal":
                text_lines = []
                if not person['has_helmet']:
                    text_lines.append(("BARET YOK!", (0, 0, 255)))
                if not person['has_vest']:
                    text_lines.append(("YELEK YOK!", (0, 0, 255)))
                
                if text_lines:
                    text_y = y1 - 10
                    for text, color in text_lines:
                        cv2.putText(annotated_frame, text, (x1, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        text_y -= 25
            else:
                # Normal/Full modda detaylı göster
                text_y = y1 - 35
                cv2.rectangle(annotated_frame, (x1, text_y - 5), (x1 + 200, y1), (0, 0, 0), -1)
                cv2.putText(annotated_frame, f"Baret: {helmet_status}", (x1 + 5, text_y + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, helmet_color, 1)
                cv2.putText(annotated_frame, f"Yelek: {vest_status}", (x1 + 5, text_y + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, vest_color, 1)
        
        # 3. Poligon Çizimi
        if len(active_polygon) > 0:
            poly_np = np.array(active_polygon, np.int32)
            
            if self.tracking_enabled:
                # Takip modunda: Şeffaf alan + çizgi
                overlay = annotated_frame.copy()
                cv2.fillPoly(overlay, [poly_np], alert_color)
                cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
                cv2.polylines(annotated_frame, [poly_np], True, alert_color, 3)
            else:
                # Çizim modunda
                cv2.polylines(annotated_frame, [poly_np], False, (0, 255, 255), 2)
                if self.display_mode != "minimal":  # Minimal modda nokta numaralarını gösterme
                    for i, pt in enumerate(active_polygon):
                        cv2.circle(annotated_frame, pt, 7, (0, 0, 255), -1)
                        cv2.circle(annotated_frame, pt, 9, (255, 255, 255), 2)
                        cv2.putText(annotated_frame, str(i+1), (pt[0] + 15, pt[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 4. Üst bilgi paneli (Minimal modda küçük)
        if self.display_mode == "minimal":
            # Sadece kritik uyarı ve mod
            if risk_level == "KRITIK":
                cv2.rectangle(annotated_frame, (10, 10), (400, 50), (0, 0, 0), -1)
                cv2.putText(annotated_frame, alert_msg, (20, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
        else:
            # Normal panel
            cv2.rectangle(annotated_frame, (10, 25), (500, 85), (0, 0, 0), -1)
            cv2.putText(annotated_frame, alert_msg, (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, alert_color, 3)
        
        # 5. Mod göstergesi (Sağ üst köşe - her zaman)
        mode_text = "LOCKED" if self.tracking_enabled else "DRAWING"
        mode_color = (0, 255, 0) if self.tracking_enabled else (0, 165, 255)
        cv2.putText(annotated_frame, mode_text, (frame_width - 130, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        
        # 6. Kişi sayısı (Sadece normal/full modda)
        if self.display_mode in ["normal", "full"] and len(persons_in_danger) > 0:
            cv2.putText(annotated_frame, f"Tehlikeli: {len(persons_in_danger)}", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 7. Görünüm modu göstergesi (Sadece full modda)
        if self.display_mode == "full":
            cv2.putText(annotated_frame, f"Display: {self.display_mode.upper()}", 
                       (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # --- KRİTİK İHLAL KAYDI (Sadece takip modundayken) ---
        if risk_level == "KRITIK" and self.tracking_enabled:
            current_time = datetime.now().timestamp()
            if current_time - self.last_save_time > 2:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"violations/ihlal_{timestamp_str}.jpg"
                cv2.imwrite(save_path, annotated_frame)
                self.last_save_time = current_time
                print(f"📸 KRİTİK İHLAL KAYDEDİLDİ: {save_path}")

        raw_data = {
            "yolo_results": yolo_results,
            "persons_in_danger": persons_in_danger,
            "risk_level": risk_level,
            "tracking_active": self.tracking_enabled
        }
        
        return annotated_frame, raw_data