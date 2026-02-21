"""
BioEntity - Data structure for tracked biological organisms
Stores position, velocity, history, and supports ghost predictions
"""
import numpy as np
from collections import deque
from typing import Tuple, Optional


class BioEntity:
    """
    Represents a single tracked biological entity (cell/organism).
    
    Attributes:
        entity_id (int): Unique identifier for this entity
        centroid_history (deque): Fixed-size queue of past centroids for trail visualization
        velocity (tuple): Current velocity vector (vx, vy) in pixels per frame
        current_bbox (tuple): Current bounding box (x1, y1, x2, y2)
        frames_missing (int): Counter for frames where entity wasn't detected
        confidence (float): Detection confidence score
        ghost_position (tuple): Predicted future position for visualization
    """
    
    def __init__(self, entity_id: int, bbox: Tuple[float, float, float, float], 
                 confidence: float = 1.0, max_history: int = 30):
        """
        Initialize a new BioEntity.
        
        Args:
            entity_id: Unique identifier for this entity
            bbox: Initial bounding box (x1, y1, x2, y2)
            confidence: Detection confidence score
            max_history: Maximum number of centroid points to retain for trail
        """
        self.entity_id = entity_id
        self.current_bbox = bbox
        self.confidence = confidence
        self.frames_missing = 0
        
        # Calculate initial centroid
        centroid = self._calculate_centroid(bbox)
        self.centroid_history = deque(maxlen=max_history)
        self.centroid_history.append(centroid)
        
        # Initialize velocity as zero
        self.velocity = (0.0, 0.0)
        
        # Additional tracking metrics
        self.total_frames_tracked = 1
        self.creation_frame = 0
        
        # Ghost prediction
        self.ghost_position = None
        
    def _calculate_centroid(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Calculate centroid from bounding box."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return (cx, cy)
    
    @property
    def centroid(self) -> Tuple[float, float]:
        """Get the most recent centroid position."""
        return self.centroid_history[-1]
    
    def update(self, bbox: Tuple[float, float, float, float], confidence: float = 1.0):
        """
        Update entity with new detection.
        
        Args:
            bbox: New bounding box (x1, y1, x2, y2)
            confidence: Detection confidence score
        """
        # Calculate new centroid
        new_centroid = self._calculate_centroid(bbox)
        
        # Calculate velocity if we have previous position
        if len(self.centroid_history) > 0:
            prev_centroid = self.centroid_history[-1]
            self.velocity = (
                new_centroid[0] - prev_centroid[0],
                new_centroid[1] - prev_centroid[1]
            )
        
        # Update state
        self.centroid_history.append(new_centroid)
        self.current_bbox = bbox
        self.confidence = confidence
        self.frames_missing = 0
        self.total_frames_tracked += 1
    
    def mark_missing(self):
        """Mark this entity as missing in the current frame."""
        self.frames_missing += 1
    
    def predict_next_position(self) -> Tuple[float, float]:
        """
        Predict the next centroid position based on current velocity.
        Useful for tracking during temporary occlusions.
        
        Returns:
            Predicted (x, y) centroid position
        """
        current = self.centroid
        predicted = (
            current[0] + self.velocity[0],
            current[1] + self.velocity[1]
        )
        self.ghost_position = predicted
        return predicted
    
    def predict_position_at_time(self, frames_ahead: int) -> Tuple[float, float]:
        """
        Predict position N frames into the future (Ghost Position).
        
        Args:
            frames_ahead: Number of frames to predict ahead
            
        Returns:
            Predicted (x, y) centroid position
        """
        current = self.centroid
        return (
            current[0] + self.velocity[0] * frames_ahead,
            current[1] + self.velocity[1] * frames_ahead
        )
    
    def get_ghost_bbox(self, frames_ahead: int = 1) -> Tuple[float, float, float, float]:
        """
        Get predicted bounding box for ghost visualization.
        
        Args:
            frames_ahead: Number of frames to predict ahead
            
        Returns:
            Predicted bbox (x1, y1, x2, y2)
        """
        # Get current bbox dimensions
        x1, y1, x2, y2 = self.current_bbox
        width = x2 - x1
        height = y2 - y1
        
        # Predict center position
        ghost_center = self.predict_position_at_time(frames_ahead)
        
        # Create bbox around predicted center
        ghost_x1 = ghost_center[0] - width / 2
        ghost_y1 = ghost_center[1] - height / 2
        ghost_x2 = ghost_center[0] + width / 2
        ghost_y2 = ghost_center[1] + height / 2
        
        return (ghost_x1, ghost_y1, ghost_x2, ghost_y2)
    
    def get_speed(self) -> float:
        """
        Calculate current speed (magnitude of velocity).
        
        Returns:
            Speed in pixels per frame
        """
        return np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
    
    def get_trail_points(self) -> list:
        """
        Get all centroid history points for trail visualization.
        
        Returns:
            List of (x, y) tuples
        """
        return list(self.centroid_history)
    
    def __repr__(self):
        return (f"BioEntity(id={self.entity_id}, "
                f"centroid={self.centroid}, "
                f"velocity={self.velocity}, "
                f"confidence={self.confidence:.2f})")
