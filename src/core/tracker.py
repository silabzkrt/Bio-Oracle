"""
CentroidTracker - Implements centroid-based tracking algorithm for biological entities
"""
import numpy as np
from scipy.spatial import distance as dist
from typing import List, Tuple, Dict, Optional
import logging
from src.core.bio_entity import BioEntity


class CentroidTracker:
    """
    Centroid-based tracking algorithm for maintaining entity persistence across frames.
    
    Matches new detections to existing entities using Euclidean distance between centroids.
    Optimized for real-time performance with single-celled organisms.
    """
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 50.0):
        """
        Initialize the centroid tracker.
        
        Args:
            max_disappeared: Maximum frames an entity can be missing before removal
            max_distance: Maximum distance (pixels) for matching detections to entities
        """
        self.next_entity_id = 0
        self.entities: Dict[int, BioEntity] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.total_entities_created = 0
        self.total_entities_destroyed = 0
        self.current_frame = 0
    
    def register(self, bbox: Tuple[float, float, float, float], 
                 confidence: float = 1.0) -> BioEntity:
        """
        Register a new entity.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            confidence: Detection confidence
        
        Returns:
            Newly created BioEntity
        """
        entity = BioEntity(self.next_entity_id, bbox, confidence)
        entity.creation_frame = self.current_frame
        
        self.entities[self.next_entity_id] = entity
        self.next_entity_id += 1
        self.total_entities_created += 1
        
        self.logger.debug(f"Registered new entity: {entity.entity_id}")
        return entity
    
    def deregister(self, entity_id: int):
        """
        Remove an entity from tracking.
        
        Args:
            entity_id: ID of entity to remove
        """
        if entity_id in self.entities:
            self.logger.debug(f"Deregistered entity: {entity_id}")
            del self.entities[entity_id]
            self.total_entities_destroyed += 1
    
    def update(self, detections: List[Tuple[float, float, float, float, float]]) -> Dict[int, BioEntity]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections [(x1, y1, x2, y2, confidence), ...]
        
        Returns:
            Dictionary of currently tracked entities {entity_id: BioEntity}
        """
        self.current_frame += 1
        
        # If no detections, mark all entities as missing
        if len(detections) == 0:
            for entity_id in list(self.entities.keys()):
                self.entities[entity_id].mark_missing()
                
                # Remove entities that have been missing too long
                if self.entities[entity_id].frames_missing > self.max_disappeared:
                    self.deregister(entity_id)
            
            return self.entities
        
        # Calculate centroids for new detections
        detection_centroids = []
        for det in detections:
            x1, y1, x2, y2, conf = det
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            detection_centroids.append((cx, cy))
        
        detection_centroids = np.array(detection_centroids)
        
        # If no existing entities, register all detections as new
        if len(self.entities) == 0:
            for i, det in enumerate(detections):
                x1, y1, x2, y2, conf = det
                self.register((x1, y1, x2, y2), conf)
        else:
            # Match detections to existing entities
            entity_ids = list(self.entities.keys())
            entity_centroids = np.array([self.entities[eid].centroid for eid in entity_ids])
            
            # Calculate pairwise distances between entities and detections
            distances = dist.cdist(entity_centroids, detection_centroids)
            
            # Find optimal assignments using Hungarian algorithm (greedy approximation)
            # For real-time performance, we use a simple greedy approach
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            # Match entities to detections
            for row, col in zip(rows, cols):
                # Skip if already used
                if row in used_rows or col in used_cols:
                    continue
                
                # Skip if distance too large
                if distances[row, col] > self.max_distance:
                    continue
                
                # Update entity
                entity_id = entity_ids[row]
                x1, y1, x2, y2, conf = detections[col]
                self.entities[entity_id].update((x1, y1, x2, y2), conf)
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle unmatched entities (mark as missing)
            unused_rows = set(range(len(entity_ids))) - used_rows
            for row in unused_rows:
                entity_id = entity_ids[row]
                self.entities[entity_id].mark_missing()
                
                # Remove if missing too long
                if self.entities[entity_id].frames_missing > self.max_disappeared:
                    self.deregister(entity_id)
            
            # Handle unmatched detections (register as new entities)
            unused_cols = set(range(len(detections))) - used_cols
            for col in unused_cols:
                x1, y1, x2, y2, conf = detections[col]
                self.register((x1, y1, x2, y2), conf)
        
        return self.entities
    
    def get_entity(self, entity_id: int) -> Optional[BioEntity]:
        """
        Get a specific entity by ID.
        
        Args:
            entity_id: Entity ID to retrieve
        
        Returns:
            BioEntity if found, None otherwise
        """
        return self.entities.get(entity_id, None)
    
    def get_all_entities(self) -> Dict[int, BioEntity]:
        """
        Get all currently tracked entities.
        
        Returns:
            Dictionary of entities {entity_id: BioEntity}
        """
        return self.entities
    
    def get_entity_count(self) -> int:
        """
        Get current number of tracked entities.
        
        Returns:
            Number of active entities
        """
        return len(self.entities)
    
    def get_statistics(self) -> dict:
        """
        Get tracking statistics.
        
        Returns:
            Dictionary with tracking metrics
        """
        return {
            'current_entities': len(self.entities),
            'total_created': self.total_entities_created,
            'total_destroyed': self.total_entities_destroyed,
            'current_frame': self.current_frame,
            'max_disappeared': self.max_disappeared,
            'max_distance': self.max_distance
        }
    
    def reset(self):
        """Reset the tracker to initial state."""
        self.entities.clear()
        self.next_entity_id = 0
        self.total_entities_created = 0
        self.total_entities_destroyed = 0
        self.current_frame = 0
        self.logger.info("Tracker reset")
    
    def __repr__(self):
        return (f"CentroidTracker(entities={len(self.entities)}, "
                f"frame={self.current_frame})")
