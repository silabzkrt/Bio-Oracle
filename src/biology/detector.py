import cv2
import numpy as np
import math

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. YOLO11 detection disabled.")


class CellDetector:
    def __init__(self, use_yolo=False, model_path=None):
        """
        Initialize Cell Detector
        
        Args:
            use_yolo (bool): Use YOLO11 for detection if available
            model_path (str): Path to YOLO model file (e.g., 'assets/models/best.pt')
        """
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        self.yolo_model = None
        
        # Load YOLO model if requested and available
        if self.use_yolo and model_path:
            try:
                self.yolo_model = YOLO(model_path)
                print(f"YOLO11 model loaded from {model_path}")
            except Exception as e:
                print(f"Error loading YOLO model: {e}")
                self.use_yolo = False
        
        # Detection parameters (more permissive for better detection)
        self.min_area = 200
        self.max_area = 5000
        self.stability_threshold = 1  # Instant locking
        
        # Tracking state
        self.locked_cells = []
        self.candidate_cells = {}
        self.next_cell_id = 1
        self.removed_cells = set()
        self.cell_positions = {}
        self.cell_velocities = {}
    
    def process(self, frame):
        """
        Process frame with either YOLO11 or traditional detection
        """
        if self.use_yolo and self.yolo_model:
            return self._process_with_yolo(frame)
        else:
            return self._process_traditional(frame)
    
    def _process_with_yolo(self, frame):
        """Process frame using YOLO11 detection"""
        output_frame = frame.copy()
        
        # Run YOLO inference
        results = self.yolo_model(frame, verbose=False, conf=0.3)
        
        # Extract detections
        current_frame_cells = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    
                    x, y = int(x1), int(y1)
                    w, h = int(x2 - x1), int(y2 - y1)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    area = w * h
                    
                    current_frame_cells.append((x, y, w, h, cx, cy, area, conf))
        
        # Update locked cells using the same tracking logic
        updated_locked_cells = []
        for locked_x, locked_y, locked_w, locked_h, cell_id in self.locked_cells:
            if cell_id not in self.cell_positions:
                self.cell_positions[cell_id] = [(locked_x + locked_w // 2, locked_y + locked_h // 2)]
            
            predicted_cx = locked_x + locked_w // 2
            predicted_cy = locked_y + locked_h // 2
            
            if cell_id in self.cell_velocities:
                vx, vy = self.cell_velocities[cell_id]
                predicted_cx += int(vx)
                predicted_cy += int(vy)
            
            found_center = None
            min_distance = float('inf')
            search_radius = 100
            
            for x, y, w, h, cx, cy, area, conf in current_frame_cells:
                dist = math.sqrt((cx - predicted_cx)**2 + (cy - predicted_cy)**2)
                if dist < min_distance and dist < search_radius:
                    area_ratio = min(area, locked_w * locked_h) / max(area, locked_w * locked_h)
                    if area_ratio > 0.6:
                        min_distance = dist
                        found_center = (cx, cy, w, h)
            
            if found_center:
                new_cx, new_cy, new_w, new_h = found_center
                old_cx, old_cy = self.cell_positions[cell_id][-1]
                
                vx = (new_cx - old_cx)
                vy = (new_cy - old_cy)
                
                if cell_id in self.cell_velocities:
                    old_vx, old_vy = self.cell_velocities[cell_id]
                    vx = 0.5 * old_vx + 0.5 * vx
                    vy = 0.5 * old_vy + 0.5 * vy
                
                self.cell_velocities[cell_id] = (vx, vy)
                self.cell_positions[cell_id].append((new_cx, new_cy))
                
                if len(self.cell_positions[cell_id]) > 5:
                    self.cell_positions[cell_id].pop(0)
                
                new_x = new_cx - new_w // 2
                new_y = new_cy - new_h // 2
                updated_locked_cells.append((new_x, new_y, new_w, new_h, cell_id))
            else:
                self.removed_cells.add(cell_id)
                if cell_id in self.cell_velocities:
                    del self.cell_velocities[cell_id]
                if cell_id in self.cell_positions:
                    del self.cell_positions[cell_id]
                print(f"Cell #{cell_id} REMOVED (left frame)")
        
        self.locked_cells = updated_locked_cells
        
        # Lock new stable detections
        current_candidates = {}
        candidate_count = 0
        
        for x, y, w, h, cx, cy, area, conf in current_frame_cells:
            cell_key = f"{cx}_{cy}"
            
            near_locked = False
            for lx, ly, lw, lh, _ in self.locked_cells:
                lcx = lx + lw // 2
                lcy = ly + lh // 2
                ldist = math.sqrt((cx - lcx)**2 + (cy - lcy)**2)
                if ldist < 40:
                    near_locked = True
                    break
            
            if near_locked:
                continue
            
            matched = False
            for key, (count, prev_x, prev_y, prev_w, prev_h) in list(self.candidate_cells.items()):
                prev_cx = prev_x + prev_w // 2
                prev_cy = prev_y + prev_h // 2
                distance = math.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                
                if distance < 40:
                    current_candidates[key] = (count + 1, x, y, w, h)
                    matched = True
                    
                    if count + 1 >= self.stability_threshold:
                        while self.next_cell_id in self.removed_cells:
                            self.next_cell_id += 1
                        
                        self.locked_cells.append((x, y, w, h, self.next_cell_id))
                        self.cell_positions[self.next_cell_id] = [(cx, cy)]
                        self.cell_velocities[self.next_cell_id] = (0, 0)
                        print(f"Cell #{self.next_cell_id} DETECTED (YOLO) at ({x}, {y}) conf={conf:.2f}")
                        self.next_cell_id += 1
                    else:
                        candidate_count += 1
                    break
            
            if not matched:
                current_candidates[cell_key] = (1, x, y, w, h)
                candidate_count += 1
        
        self.candidate_cells = current_candidates
        
        # Draw YOLO detections
        for locked_x, locked_y, locked_w, locked_h, cell_id in self.locked_cells:
            cv2.rectangle(output_frame, (locked_x, locked_y), 
                         (locked_x + locked_w, locked_y + locked_h), (0, 255, 0), 3)
            
            label = f"#{cell_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(output_frame, (locked_x, locked_y-label_size[1]-10), 
                          (locked_x+label_size[0]+6, locked_y), (0, 255, 0), -1)
            cv2.putText(output_frame, label, (locked_x+3, locked_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            locked_cx = locked_x + locked_w // 2
            locked_cy = locked_y + locked_h // 2
            cv2.circle(output_frame, (locked_cx, locked_cy), 5, (0, 0, 255), -1)
        
        # Draw candidates
        for x, y, w, h, cx, cy, area, conf in current_frame_cells:
            is_locked = False
            for lx, ly, lw, lh, _ in self.locked_cells:
                if abs(cx - (lx + lw//2)) < 20 and abs(cy - (ly + lh//2)) < 20:
                    is_locked = True
                    break
            
            if not is_locked:
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.putText(output_frame, f"{conf:.2f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Info overlay
        cv2.rectangle(output_frame, (10, 10), (400, 110), (0, 0, 0), -1)
        cv2.putText(output_frame, "YOLO11 Detection", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(output_frame, f"Locked: {len(self.locked_cells)}", (20, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output_frame, f"Candidates: {candidate_count}", (20, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return output_frame, len(self.locked_cells), candidate_count
    
    def _process_traditional(self, frame):
        """Process frame using basic OpenCV detection"""
        output_frame = frame.copy()
        
        # Simple preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Simple threshold
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Detect cells
        detected_cells = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    detected_cells.append((x, y, w, h, cx, cy))
        
        # Lock cells (first time detection)
        if len(self.locked_cells) == 0:
            for x, y, w, h, cx, cy in detected_cells:
                self.locked_cells.append((x, y, w, h, self.next_cell_id))
                print(f"Cell #{self.next_cell_id} LOCKED at ({x}, {y})")
                self.next_cell_id += 1
        
        # Draw locked cells
        for locked_x, locked_y, locked_w, locked_h, cell_id in self.locked_cells:
            cv2.rectangle(output_frame, (locked_x, locked_y), 
                         (locked_x + locked_w, locked_y + locked_h), (255, 0, 0), 3)
            
            label = f"#{cell_id}"
            cv2.putText(output_frame, label, (locked_x, locked_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Info
        cv2.rectangle(output_frame, (10, 10), (250, 50), (0, 0, 0), -1)
        cv2.putText(output_frame, f"Locked: {len(self.locked_cells)}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return output_frame, len(self.locked_cells), 0
    
    def get_cell_count(self):
        """
        Get the current count of locked (detected and tracked) cells
        
        Returns:
            int: Number of locked cells
        """
        return len(self.locked_cells)
    

if __name__ == "__main__":
    video_path = "video2.mp4"
    output_file = "video_test_results.txt"
    
    cap = cv2.VideoCapture(video_path)
    detector = CellDetector()

    if not cap.isOpened():
        print("Error: Video file not found!")
        exit()

    print("Processing video... Press 'q' to quit.")
    
    with open(output_file, 'w') as f:
        f.write("Video Tracking Test Results\n")
        f.write("=" * 60 + "\n\n")
        
        frame_number = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video ended.")
                f.write(f"\nVideo processing completed at frame {frame_number}\n")
                break

            frame = cv2.resize(frame, (1024, 768)) 

            processed_frame, locked_count, candidate_count = detector.process(frame)
            
            f.write(f"Frame {frame_number}: {locked_count} locked, {candidate_count} candidates\n")
            
            cv2.imshow("Bio-Oracle: Cell Tracking System", processed_frame)

            frame_number += 1
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                f.write(f"\nUser stopped at frame {frame_number}\n")
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Results saved to {output_file}")