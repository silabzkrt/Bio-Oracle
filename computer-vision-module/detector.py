import cv2
import numpy as np
import math


class CellDetector:
    def __init__(self):
        self.min_area = 300
        self.max_area = 20000
        self.locked_cells = []  # Store locked cells: (x, y, w, h, cell_id)
        self.next_cell_id = 1
        self.candidate_cells = {}  # Track potential new cells across frames
        self.stability_threshold = 50
        
    def process(self, frame):
        output_frame = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Mask out locked cells from the binary image
        search_mask = closing.copy()
        for locked_x, locked_y, locked_w, locked_h, _ in self.locked_cells:
            # Create a larger exclusion zone to avoid re-detecting near locked cells
            padding = 10
            x1 = max(0, locked_x - padding)
            y1 = max(0, locked_y - padding)
            x2 = min(search_mask.shape[1], locked_x + locked_w + padding)
            y2 = min(search_mask.shape[0], locked_y + locked_h + padding)
            cv2.rectangle(search_mask, (x1, y1), (x2, y2), 0, -1)
        
        contours, _ = cv2.findContours(search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_cells = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                
                aspect_ratio = float(w) / h if h > 0 else 0
                if 0.2 < aspect_ratio < 6.0:
                    
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        if circularity > 0.15:
                            M = cv2.moments(cnt)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                valid_cells.append((cnt, x, y, w, h, cx, cy, area))
        
        valid_cells.sort(key=lambda x: x[7], reverse=True)
        
        filtered_cells = []
        for cell in valid_cells:
            cnt, x, y, w, h, cx, cy, area = cell
            
            is_duplicate = False
            for existing in filtered_cells:
                ex_x, ex_y, ex_w, ex_h = existing[1], existing[2], existing[3], existing[4]
                ex_cx, ex_cy = existing[5], existing[6]
                ex_area = existing[7]
                
                # Calculate intersection area
                x1_inter = max(x, ex_x)
                y1_inter = max(y, ex_y)
                x2_inter = min(x + w, ex_x + ex_w)
                y2_inter = min(y + h, ex_y + ex_h)
                
                inter_width = max(0, x2_inter - x1_inter)
                inter_height = max(0, y2_inter - y1_inter)
                intersection_area = inter_width * inter_height
                
                current_area = w * h
                
                # Calculate IoU (Intersection over Union)
                union_area = current_area + ex_area - intersection_area
                iou = intersection_area / union_area if union_area > 0 else 0
                
                # Check if current cell is inside existing cell
                inside_existing = (x >= ex_x and y >= ex_y and 
                                 x + w <= ex_x + ex_w and y + h <= ex_y + ex_h)
                
                # Check if existing cell is inside current cell
                existing_inside_current = (ex_x >= x and ex_y >= y and 
                                          ex_x + ex_w <= x + w and ex_y + ex_h <= y + h)
                
                # Check center distance
                distance = math.sqrt((cx - ex_cx)**2 + (cy - ex_cy)**2)
                
                # Mark as duplicate if:
                # 1. High IoU (significant overlap)
                # 2. Current cell is completely inside an existing cell
                # 3. Centers are very close
                # 4. Intersection covers more than 40% of the smaller cell
                if (iou > 0.3 or 
                    inside_existing or 
                    distance < 30 or 
                    intersection_area > 0.4 * min(current_area, ex_area)):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_cells.append(cell)
        
        # Track candidates across frames and lock stable cells
        current_candidates = {}
        for cell in filtered_cells:
            cnt, x, y, w, h, cx, cy, area = cell
            cell_key = f"{cx}_{cy}"
            
            # Check if this matches any existing candidate
            matched = False
            for key, (count, prev_x, prev_y, prev_w, prev_h) in list(self.candidate_cells.items()):
                prev_cx = prev_x + prev_w // 2
                prev_cy = prev_y + prev_h // 2
                distance = math.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                
                if distance < 50:  # Same cell if center moved less than 50 pixels
                    current_candidates[key] = (count + 1, x, y, w, h)
                    matched = True
                    
                    # Lock this cell if it's been stable enough
                    if count + 1 >= self.stability_threshold:
                        # Check if not already locked
                        already_locked = False
                        for lx, ly, lw, lh, _ in self.locked_cells:
                            lcx = lx + lw // 2
                            lcy = ly + lh // 2
                            ldist = math.sqrt((cx - lcx)**2 + (cy - lcy)**2)
                            if ldist < 50:
                                already_locked = True
                                break
                        
                        if not already_locked:
                            self.locked_cells.append((x, y, w, h, self.next_cell_id))
                            print(f"Cell #{self.next_cell_id} DETECTED at ({x}, {y})")
                            self.next_cell_id += 1
                    break
            
            if not matched:
                # New candidate
                current_candidates[cell_key] = (1, x, y, w, h)
        
        self.candidate_cells = current_candidates
        
        mask_visual = cv2.cvtColor(search_mask, cv2.COLOR_GRAY2BGR)
        
        # Draw locked cells first (in blue to show they're locked)
        for locked_x, locked_y, locked_w, locked_h, cell_id in self.locked_cells:
            cv2.rectangle(output_frame, (locked_x, locked_y), (locked_x + locked_w, locked_y + locked_h), (255, 0, 0), 3)
            cv2.rectangle(mask_visual, (locked_x, locked_y), (locked_x + locked_w, locked_y + locked_h), (255, 0, 0), 3)
            
            # Corner markers
            cv2.circle(output_frame, (locked_x, locked_y), 5, (255, 0, 0), -1)
            cv2.circle(output_frame, (locked_x + locked_w, locked_y), 5, (255, 0, 0), -1)
            cv2.circle(output_frame, (locked_x, locked_y + locked_h), 5, (255, 0, 0), -1)
            cv2.circle(output_frame, (locked_x + locked_w, locked_y + locked_h), 5, (255, 0, 0), -1)
            
            # Locked label
            label = f"#{cell_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(output_frame, (locked_x, locked_y-label_size[1]-10), (locked_x+label_size[0]+6, locked_y), (255, 0, 0), -1)
            cv2.putText(output_frame, label, (locked_x+3, locked_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(mask_visual, f"#{cell_id} LOCKED", (locked_x+3, locked_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Center point
            locked_cx = locked_x + locked_w // 2
            locked_cy = locked_y + locked_h // 2
            cv2.circle(output_frame, (locked_cx, locked_cy), 5, (0, 0, 255), -1)
        
        # Draw candidate cells (in green - not yet locked)
        candidate_count = 0
        for cell in filtered_cells:
            cnt, x, y, w, h, cx, cy, area = cell
            candidate_count += 1
            
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(mask_visual, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            cv2.circle(output_frame, (x, y), 5, (255, 0, 0), -1)
            cv2.circle(output_frame, (x + w, y), 5, (255, 0, 0), -1)
            cv2.circle(output_frame, (x, y + h), 5, (255, 0, 0), -1)
            cv2.circle(output_frame, (x + w, y + h), 5, (255, 0, 0), -1)
            
            label = f"Waiting..."
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(output_frame, (x, y-label_size[1]-10), (x+label_size[0]+6, y), (0, 255, 0), -1)
            cv2.putText(output_frame, label, (x+3, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            cv2.putText(mask_visual, "Candidate", (x+3, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.circle(output_frame, (cx, cy), 5, (0, 0, 255), -1)
        
        cv2.rectangle(output_frame, (10, 10), (320, 85), (0, 0, 0), -1)
        cv2.putText(output_frame, f"Locked: {len(self.locked_cells)}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(output_frame, f"Candidates: {candidate_count}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.rectangle(mask_visual, (10, 10), (180, 55), (0, 0, 0), -1)
        cv2.putText(mask_visual, "Mask View", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        split_screen = np.hstack((output_frame, mask_visual))

        return split_screen, len(self.locked_cells), candidate_count
    

if __name__ == "__main__":
    video_path = "test_video.mp4"
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