"""
VisionManager - Unified cell detection module
Supports both YOLOv11 (ML-based) and Traditional CV methods (blob, contour, edge detection)
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class VisionManager:
    """
    Unified vision manager that supports multiple detection methods:
    - YOLOv11: Deep learning-based detection (good for general objects)
    - Traditional CV: Better for microscopy cells (blob, contour, edge detection)
    """
    
    def __init__(self, detection_method: str = 'traditional',
                 model_path: str = "yolo11n.pt",
                 conf_threshold: float = 0.5,
                 device: str = 'cpu',
                 # Traditional CV parameters
                 min_area: int = 400,
                 max_area: int = 25000,
                 blur_kernel: int = 5):
        """
        Initialize VisionManager with specified detection method.
        
        Args:
            detection_method: 'yolo' or 'traditional'
            model_path: Path to YOLO model (if using YOLO)
            conf_threshold: Confidence threshold for detections
            device: Device for YOLO ('cpu', 'cuda', 'mps')
            min_area: Min cell area for traditional methods
            max_area: Max cell area for traditional methods
            blur_kernel: Gaussian blur kernel size
        """
        self.detection_method = detection_method
        self.conf_threshold = conf_threshold
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # YOLO setup (if requested)
        self.model = None
        if detection_method == 'yolo':
            if not YOLO_AVAILABLE:
                self.logger.warning("YOLO not available, falling back to traditional CV")
                self.detection_method = 'traditional'
            else:
                self._init_yolo(model_path)
        
        # Traditional CV setup
        self.min_area = min_area
        self.max_area = max_area
        self.blur_kernel = blur_kernel
        self._init_traditional_detector()
        
        # Performance tracking
        self.frame_count = 0
        
        self.logger.info(f"VisionManager initialized (method: {self.detection_method})")
    
    def _init_yolo(self, model_path: str):
        """Initialize YOLO model."""
        try:
            self.logger.info(f"Loading YOLOv11 from {model_path}...")
            self.model = YOLO(model_path)
            self.model.to(self.device)
            self.logger.info(f"YOLO loaded on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO: {e}")
            self.detection_method = 'traditional'
    
    def _init_traditional_detector(self):
        """Initialize traditional CV blob detector."""
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = self.min_area
        params.maxArea = self.max_area
        params.filterByCircularity = True
        params.minCircularity = 0.03
        params.filterByConvexity = True
        params.minConvexity = 0.3
        params.filterByInertia = True
        params.minInertiaRatio = 0.15
        
        self.blob_detector = cv2.SimpleBlobDetector_create(params)
    
    def detect(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """
        Detect cells in frame using configured method.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            List of detections [(x1, y1, x2, y2, confidence), ...]
        """
        self.frame_count += 1
        
        if self.detection_method == 'yolo' and self.model:
            return self._detect_yolo(frame)
        else:
            return self._detect_traditional(frame, method='contours')
    
    def _detect_yolo(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """YOLO-based detection."""
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device
        )
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = box
                detections.append((float(x1), float(y1), float(x2), float(y2), float(conf)))
        
        return detections
    
    def _detect_traditional(self, frame: np.ndarray, 
                          method: str = 'contours') -> List[Tuple[float, float, float, float, float]]:
        """Traditional CV detection (contours with watershed)."""
        if method == 'contours':
            return self._detect_contours_watershed(frame)
        elif method == 'edges':
            return self._detect_edges(frame)
        else:
            return self._detect_blobs(frame)
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for detection."""
        # Grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        
        # Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        return enhanced
    
    def _detect_contours_watershed(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """
        Detect cells using contours with watershed segmentation.
        Best method for separating touching/overlapping cells.
        """
        processed = self._preprocess_frame(frame)
        
        # Otsu thresholding (inverted for dark cells)
        _, binary = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=2)
        
        # Sure background (dilate)
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        sure_bg = cv2.dilate(binary, kernel_large, iterations=3)
        
        # Sure foreground (distance transform)
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Watershed
        frame_color = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(frame_color, markers)
        
        # Extract contours from watershed results
        binary_result = np.zeros_like(processed)
        binary_result[markers > 1] = 255
        
        contours, _ = cv2.findContours(binary_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_area or area > self.max_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            if w < 10 or h < 10:
                continue
            
            # Shape properties
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.03:
                continue
            
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio < 0.08 or aspect_ratio > 12.0:
                continue
            
            confidence = min(circularity * 2.0, 1.0)
            detections.append((float(x), float(y), float(x + w), float(y + h), confidence))
        
        return detections
    
    def _detect_edges(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """Detect cells with dark outlines using edge detection."""
        processed = self._preprocess_frame(frame)
        
        # Canny edge detection
        edges = cv2.Canny(processed, 10, 40)
        
        # Dilate to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        edges = cv2.dilate(edges, kernel, iterations=3)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_area or area > self.max_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            if w < 10 or h < 10:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.08:
                continue
            
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                continue
            
            confidence = min(circularity * 2.0, 1.0)
            detections.append((float(x), float(y), float(x + w), float(y + h), confidence))
        
        return detections
    
    def _detect_blobs(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """Detect cells using blob detection."""
        processed = self._preprocess_frame(frame)
        inverted = cv2.bitwise_not(processed)
        
        keypoints = self.blob_detector.detect(inverted)
        
        detections = []
        for kp in keypoints:
            x, y = kp.pt
            size = kp.size
            
            x1 = x - size / 2
            y1 = y - size / 2
            x2 = x + size / 2
            y2 = y + size / 2
            
            confidence = min(kp.response, 1.0)
            detections.append((float(x1), float(y1), float(x2), float(y2), confidence))
        
        return detections
    
    def set_detection_method(self, method: str):
        """Switch detection method at runtime."""
        if method in ['yolo', 'traditional']:
            self.detection_method = method
            self.logger.info(f"Switched to {method} detection")
    
    def get_stats(self) -> dict:
        """Get detection statistics."""
        return {
            'method': self.detection_method,
            'frames_processed': self.frame_count,
            'min_area': self.min_area,
            'max_area': self.max_area
        }
