from ultralytics import YOLO
import numpy as np

class CellDetector:
    def __init__(self, model_path, confidence_threshold=0.5, device='cpu'):
        self.model = YOLO(model_path)
        self.conf = confidence_threshold
        self.device = device

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, device=self.device, verbose=False)
        detections = []
        
        for r in results[0].boxes:
            box = r.xyxy[0].cpu().numpy().astype(int)
            detections.append({
                'bbox': box,
                'confidence': float(r.conf)
            })
        return detections