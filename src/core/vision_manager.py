"""
VisionManager - Handles YOLOv11 model loading and inference for cell detection
"""
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional
import logging


class VisionManager:
    """
    Manages computer vision operations using YOLOv11 for organism detection.
    
    Optimized for real-time performance (30+ FPS).
    """
    
    def __init__(self, model_path: str = "yolo11n.pt", conf_threshold: float = 0.5, 
                 iou_threshold: float = 0.45, device: str = 'cpu'):
        """
        Initialize the YOLOv11 vision manager.
        
        Args:
            model_path: Path to YOLO model weights (yolo11n.pt, yolo11s.pt, etc.)
            conf_threshold: Confidence threshold for detections (0.0-1.0)
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ('cpu', 'cuda', 'mps')
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Load YOLO model
        self.logger.info(f"Loading YOLOv11 model from {model_path}...")
        try:
            self.model = YOLO(model_path)
            self.model.to(device)
            self.logger.info(f"Model loaded successfully on {device}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
        
        # Performance tracking
        self.frame_count = 0
        self.total_inference_time = 0.0
    
    def detect(self, frame: np.ndarray, target_classes: Optional[List[int]] = None) -> List[Tuple]:
        """
        Perform object detection on a single frame.
        
        Args:
            frame: Input image/frame (numpy array)
            target_classes: List of class IDs to detect (None = all classes)
        
        Returns:
            List of detections, each as tuple: (x1, y1, x2, y2, confidence, class_id)
        """
        # Run inference
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device,
            stream=False
        )
        
        detections = []
        
        # Parse results
        if len(results) > 0:
            result = results[0]  # Single image inference
            
            # Extract bounding boxes, confidences, and class IDs
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    # Filter by target classes if specified
                    if target_classes is None or int(cls_id) in target_classes:
                        x1, y1, x2, y2 = box
                        detections.append((
                            float(x1), float(y1), float(x2), float(y2),
                            float(conf), int(cls_id)
                        ))
        
        self.frame_count += 1
        return detections
    
    def detect_cells(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """
        Convenience method for cell detection (returns simplified format).
        
        Args:
            frame: Input image/frame
        
        Returns:
            List of detections: [(x1, y1, x2, y2, confidence), ...]
        """
        detections = self.detect(frame)
        # Return only bbox and confidence, drop class_id
        return [(x1, y1, x2, y2, conf) for x1, y1, x2, y2, conf, _ in detections]
    
    def get_average_inference_time(self) -> float:
        """
        Calculate average inference time per frame.
        
        Returns:
            Average time in milliseconds
        """
        if self.frame_count == 0:
            return 0.0
        return (self.total_inference_time / self.frame_count) * 1000
    
    def update_thresholds(self, conf: Optional[float] = None, iou: Optional[float] = None):
        """
        Update detection thresholds dynamically.
        
        Args:
            conf: New confidence threshold
            iou: New IoU threshold
        """
        if conf is not None:
            self.conf_threshold = conf
            self.logger.info(f"Updated confidence threshold to {conf}")
        
        if iou is not None:
            self.iou_threshold = iou
            self.logger.info(f"Updated IoU threshold to {iou}")
    
    def warmup(self, frame_shape: Tuple[int, int, int] = (640, 640, 3), iterations: int = 3):
        """
        Warm up the model to optimize first-frame performance.
        
        Args:
            frame_shape: Shape of dummy frame (height, width, channels)
            iterations: Number of warmup iterations
        """
        self.logger.info("Warming up model...")
        dummy_frame = np.zeros(frame_shape, dtype=np.uint8)
        
        for i in range(iterations):
            _ = self.detect(dummy_frame)
        
        self.logger.info(f"Warmup complete ({iterations} iterations)")
        
        # Reset counters after warmup
        self.frame_count = 0
        self.total_inference_time = 0.0
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'model_type': 'YOLOv11',
            'device': self.device,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'frames_processed': self.frame_count
        }
    
    def __repr__(self):
        return (f"VisionManager(device={self.device}, "
                f"conf={self.conf_threshold}, "
                f"frames={self.frame_count})")
