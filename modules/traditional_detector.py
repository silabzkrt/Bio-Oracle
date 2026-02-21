"""
Traditional Computer Vision Cell Detector
Uses blob detection and contour methods to detect ALL cells in each frame
Better for microscopy images than YOLO
"""
import cv2
import numpy as np
from typing import List, Tuple
import logging


class TraditionalCellDetector:
    """
    Detects cells using traditional CV methods (blob detection, contours).
    Processes each frame independently - no ML model needed.
    """
    
    def __init__(self, min_area=100, max_area=50000, 
                 blur_kernel=5, threshold_method='adaptive'):
        """
        Initialize traditional cell detector.
        
        Args:
            min_area: Minimum cell area in pixels
            max_area: Maximum cell area in pixels
            blur_kernel: Gaussian blur kernel size (odd number)
            threshold_method: 'adaptive', 'otsu', or 'simple'
        """
        self.min_area = min_area
        self.max_area = max_area
        self.blur_kernel = blur_kernel
        self.threshold_method = threshold_method
        self.logger = logging.getLogger(__name__)
        
        # Setup blob detector
        params = cv2.SimpleBlobDetector_Params()
        
        # Filter by area
        params.filterByArea = True
        params.minArea = min_area
        params.maxArea = max_area
        
        # Filter by circularity (cells are roundish)
        params.filterByCircularity = True
        params.minCircularity = 0.4  # Balanced - allows slightly irregular cells
        
        # Filter by convexity
        params.filterByConvexity = True
        params.minConvexity = 0.6  # Moderate requirement
        
        # Filter by inertia (roundness)
        params.filterByInertia = True
        params.minInertiaRatio = 0.3  # Balanced roundness
        
        self.blob_detector = cv2.SimpleBlobDetector_create(params)
        
        self.logger.info(f"TraditionalCellDetector initialized (area: {min_area}-{max_area})")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for better cell detection.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        return enhanced
    
    def detect_with_edges(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """
        Detect cells with dark outlines using edge detection.
        Best for cells with visible dark edges/borders, including elongated cells.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detections: [(x1, y1, x2, y2, confidence), ...]
        """
        # Preprocess
        processed = self.preprocess_frame(frame)
        
        # Apply VERY sensitive Canny edge detection to find ALL dark outlines
        edges = cv2.Canny(processed, 10, 40)  # Even lower thresholds
        
        # Dilate edges MORE to connect broken outlines
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # Even larger
        edges = cv2.dilate(edges, kernel, iterations=3)  # More iterations
        
        # Close gaps in outlines
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours from edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate shape properties
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Accept ALMOST ANY shape with an outline (very low threshold)
            if circularity < 0.08:  # Extremely low - just needs to be somewhat enclosed
                continue
            
            # Very relaxed aspect ratio - allow ANY elongated cells
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:  # Very wide range
                continue
            
            # Minimum size check for width and height
            if w < 10 or h < 10:  # Too small to be a cell
                continue
            
            confidence = min(circularity * 2.0, 1.0)
            detections.append((float(x), float(y), float(x + w), float(y + h), confidence))
        
        return detections
    
    def detect_with_contours(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """
        Detect cells using contour detection with watershed segmentation.
        Optimized for dark elongated cells on light background.
        Separates touching/overlapping cells.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detections: [(x1, y1, x2, y2, confidence), ...]
        """
        # Preprocess
        processed = self.preprocess_frame(frame)
        
        # Use Otsu thresholding for better dark cell detection
        _, binary = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Remove noise with opening
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=2)
        
        # Find sure background area (dilate)
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        sure_bg = cv2.dilate(binary, kernel_large, iterations=3)
        
        # Find sure foreground area (distance transform + threshold)
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Find unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        frame_color = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(frame_color, markers)
        
        # Create binary from watershed results
        binary = np.zeros_like(processed)
        binary[markers > 1] = 255
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Minimum dimension check (smaller since watershed separates cells)
            if w < 10 or h < 10:
                continue
            
            # Calculate shape properties
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # VERY relaxed - accept almost any enclosed shape
            if circularity < 0.03:  # Even lower - watershed creates irregular shapes
                continue
            
            # Very wide aspect ratio range for elongated cells
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio < 0.08 or aspect_ratio > 12.0:  # Even wider range
                continue
            
            confidence = min(circularity * 2.0, 1.0)
            
            # Add detection
            detections.append((float(x), float(y), float(x + w), float(y + h), confidence))
        
        return detections
    
    def detect_with_blobs(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """
        Detect cells using blob detection.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detections: [(x1, y1, x2, y2, confidence), ...]
        """
        # Preprocess
        processed = self.preprocess_frame(frame)
        
        # Invert for blob detection (blobs should be bright on dark background)
        inverted = cv2.bitwise_not(processed)
        
        # Detect blobs
        keypoints = self.blob_detector.detect(inverted)
        
        detections = []
        for kp in keypoints:
            # Get blob properties
            x, y = kp.pt
            size = kp.size
            
            # Create bounding box from blob
            x1 = x - size / 2
            y1 = y - size / 2
            x2 = x + size / 2
            y2 = y + size / 2
            
            # Use blob response as confidence
            confidence = min(kp.response, 1.0)
            
            detections.append((float(x1), float(y1), float(x2), float(y2), confidence))
        
        return detections
    
    def detect_hybrid(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """
        Hybrid detection using edges, contours and blobs, then merging results.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detections: [(x1, y1, x2, y2, confidence), ...]
        """
        # Get detections from all methods
        edge_dets = self.detect_with_edges(frame)
        contour_dets = self.detect_with_contours(frame)
        blob_dets = self.detect_with_blobs(frame)
        
        # Combine and remove duplicates
        all_detections = edge_dets + contour_dets + blob_dets
        
        # Simple deduplication based on overlap
        final_detections = []
        for det in all_detections:
            x1, y1, x2, y2, conf = det
            
            # Check if overlaps with existing detection
            is_duplicate = False
            for existing in final_detections:
                ex1, ey1, ex2, ey2, econf = existing
                
                # Calculate IoU
                ix1 = max(x1, ex1)
                iy1 = max(y1, ey1)
                ix2 = min(x2, ex2)
                iy2 = min(y2, ey2)
                
                if ix1 < ix2 and iy1 < iy2:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    det_area = (x2 - x1) * (y2 - y1)
                    ex_area = (ex2 - ex1) * (ey2 - ey1)
                    iou = inter_area / (det_area + ex_area - inter_area)
                    
                    if iou > 0.5:  # Significant overlap
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                final_detections.append(det)
        
        return final_detections
    
    def detect(self, frame: np.ndarray, method='edges') -> List[Tuple[float, float, float, float, float]]:
        """
        Detect cells in a frame using specified method.
        
        Args:
            frame: Input frame
            method: 'edges', 'contours', 'blobs', or 'hybrid'
            
        Returns:
            List of detections: [(x1, y1, x2, y2, confidence), ...]
        """
        if method == 'edges':
            return self.detect_with_edges(frame)
        elif method == 'contours':
            return self.detect_with_contours(frame)
        elif method == 'blobs':
            return self.detect_with_blobs(frame)
        else:  # hybrid
            return self.detect_hybrid(frame)
    
    def visualize_preprocessing(self, frame: np.ndarray) -> np.ndarray:
        """
        Visualize preprocessing steps for debugging.
        
        Args:
            frame: Input frame
            
        Returns:
            Visualization frame showing preprocessing steps
        """
        # Get grayscale and processed versions
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        processed = self.preprocess_frame(frame)
        
        # Apply threshold
        _, binary = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Stack for comparison
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # Resize for side-by-side display
        h, w = frame.shape[:2]
        frame_resized = cv2.resize(frame, (w//2, h//2))
        gray_resized = cv2.resize(gray_bgr, (w//2, h//2))
        processed_resized = cv2.resize(processed_bgr, (w//2, h//2))
        binary_resized = cv2.resize(binary_bgr, (w//2, h//2))
        
        # Add labels
        cv2.putText(frame_resized, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(gray_resized, "Grayscale", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_resized, "Enhanced", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(binary_resized, "Binary", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Combine
        top_row = np.hstack([frame_resized, gray_resized])
        bottom_row = np.hstack([processed_resized, binary_resized])
        result = np.vstack([top_row, bottom_row])
        
        return result
