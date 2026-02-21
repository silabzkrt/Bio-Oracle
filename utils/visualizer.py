"""
Visualizer - Handles all visualization and rendering for Bio-Oracle
Draws bounding boxes, trails, ghost predictions, info panels, and more
"""
import cv2
import numpy as np
from typing import Dict, Tuple
import logging


class Visualizer:
    """
    Handles all visualization for Bio-Oracle framework.
    Draws bounding boxes, trails, ghost predictions, velocity vectors, and info panels.
    """
    
    def __init__(self, show_trails: bool = True, show_ghosts: bool = True,
                 ghost_frames_ahead: int = 10):
        """
        Initialize visualizer.
        
        Args:
            show_trails: Whether to show cell movement trails
            show_ghosts: Whether to show ghost predictions
            ghost_frames_ahead: How many frames ahead to predict ghosts
        """
        self.show_trails = show_trails
        self.show_ghosts = show_ghosts
        self.ghost_frames_ahead = ghost_frames_ahead
        self.logger = logging.getLogger(__name__)
        
        # Colors (BGR format for OpenCV)
        self.COLOR_CURRENT = (0, 255, 0)  # Green
        self.COLOR_GHOST = (255, 255, 0)  # Cyan
        self.COLOR_TRAIL = (0, 255, 0)  # Green
        self.COLOR_VELOCITY = (255, 0, 0)  # Blue
        self.COLOR_BACKGROUND = (20, 20, 20)  # Dark gray
        self.COLOR_TEXT = (255, 255, 255)  # White
        self.COLOR_HIGHLIGHT = (0, 255, 255)  # Yellow
    
    def visualize_frame(self, frame: np.ndarray, entities: Dict, 
                       population_state: Dict, prediction_text: str,
                       frame_num: int) -> np.ndarray:
        """
        Visualize complete frame with all tracking data.
        
        Args:
            frame: Input BGR frame
            entities: Dictionary of tracked entities {id: BioEntity}
            population_state: Population metrics dictionary
            prediction_text: Oracle prediction text
            frame_num: Current frame number
            
        Returns:
            Annotated frame
        """
        output = frame.copy()
        
        # Draw ghost predictions (semi-transparent layer)
        if self.show_ghosts:
            output = self._draw_ghosts(output, entities)
        
        # Draw current entities with trails
        output = self._draw_entities(output, entities)
        
        # Draw info panel
        output = self._draw_info_panel(output, len(entities), population_state,
                                       prediction_text, frame_num)
        
        return output
    
    def _draw_ghosts(self, frame: np.ndarray, entities: Dict) -> np.ndarray:
        """Draw ghost predictions (semi-transparent)."""
        ghost_overlay = frame.copy()
        
        for entity in entities.values():
            # Get ghost position
            ghost_bbox = entity.get_ghost_bbox(self.ghost_frames_ahead)
            x1, y1, x2, y2 = map(int, ghost_bbox)
            
            # Draw ghost bbox (cyan)
            cv2.rectangle(ghost_overlay, (x1, y1), (x2, y2), self.COLOR_GHOST, 2)
            cv2.circle(ghost_overlay, 
                      (int((x1+x2)/2), int((y1+y2)/2)),
                      5, self.COLOR_GHOST, -1)
            
            # Draw velocity arrow from current to ghost
            current_center = entity.centroid
            ghost_center = entity.predict_position_at_time(self.ghost_frames_ahead)
            cv2.arrowedLine(ghost_overlay,
                          (int(current_center[0]), int(current_center[1])),
                          (int(ghost_center[0]), int(ghost_center[1])),
                          self.COLOR_GHOST, 2, tipLength=0.3)
        
        # Blend ghost layer (40% opacity)
        cv2.addWeighted(ghost_overlay, 0.4, frame, 0.6, 0, frame)
        
        return frame
    
    def _draw_entities(self, frame: np.ndarray, entities: Dict) -> np.ndarray:
        """Draw current entities with trails and velocity vectors."""
        for entity_id, entity in entities.items():
            # Draw trail
            if self.show_trails and len(entity.centroid_history) > 1:
                points = entity.get_trail_points()
                for i in range(1, len(points)):
                    # Gradient effect: older points are more transparent
                    alpha = i / len(points)
                    color = (0, int(255 * alpha), 0)
                    thickness = max(1, int(2 * alpha))
                    cv2.line(frame,
                            (int(points[i-1][0]), int(points[i-1][1])),
                            (int(points[i][0]), int(points[i][1])),
                            color, thickness)
            
            # Draw current bounding box (green)
            x1, y1, x2, y2 = map(int, entity.current_bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_CURRENT, 2)
            
            # Draw centroid
            cx, cy = map(int, entity.centroid)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            
            # Draw entity ID
            cv2.putText(frame, f"ID:{entity_id}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_CURRENT, 2)
            
            # Draw velocity vector
            speed = entity.get_speed()
            if speed > 0.5:  # Only show if moving
                end_x = int(cx + entity.velocity[0] * 5)
                end_y = int(cy + entity.velocity[1] * 5)
                cv2.arrowedLine(frame, (cx, cy), (end_x, end_y),
                              self.COLOR_VELOCITY, 2, tipLength=0.4)
        
        return frame
    
    def _draw_info_panel(self, frame: np.ndarray, cell_count: int,
                        population_state: Dict, prediction_text: str,
                        frame_num: int) -> np.ndarray:
        """Draw information panel with population metrics and predictions."""
        h, w = frame.shape[:2]
        
        # Semi-transparent panel background
        panel_height = 180
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), self.COLOR_BACKGROUND, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "Bio-Oracle: Cell Tracking & Population Prediction", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_HIGHLIGHT, 2)
        
        # Current state
        y_offset = 50
        info_lines = [
            f"Frame: {frame_num} | Cells: {cell_count}",
            f"Density: {population_state.get('density_per_1000px', 0):.2f} cells/1000px²",
            f"Growth Rate: {population_state.get('growth_rate', 0):+.3f} cells/s",
            f"Per-Capita: {population_state.get('per_capita_growth_rate', 0):+.4f} /s",
        ]
        
        if 'avg_speed' in population_state:
            info_lines.append(
                f"Avg Velocity: {population_state['avg_speed']:.2f} px/frame"
            )
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, y_offset + i * 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT, 1)
        
        # Oracle prediction (highlighted)
        cv2.putText(frame, f"ORACLE: {prediction_text}",
                   (10, y_offset + len(info_lines) * 22 + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_HIGHLIGHT, 2)
        
        # Legend
        legend_y = panel_height - 25
        cv2.putText(frame, "Green=Current | Cyan=Ghost | Red=Velocity | G=Toggle Ghosts | T=Toggle Trails",
                   (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def draw_simple(self, frame: np.ndarray, entities: Dict, 
                   cell_count: int, frame_num: int) -> np.ndarray:
        """
        Simple visualization without population metrics (for faster rendering).
        
        Args:
            frame: Input frame
            entities: Tracked entities
            cell_count: Number of cells
            frame_num: Frame number
            
        Returns:
            Annotated frame
        """
        output = frame.copy()
        
        # Draw entities
        for entity_id, entity in entities.items():
            x1, y1, x2, y2 = map(int, entity.current_bbox)
            cv2.rectangle(output, (x1, y1), (x2, y2), self.COLOR_CURRENT, 2)
            
            # Draw ID
            cv2.putText(output, f"{entity_id}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_CURRENT, 1)
        
        # Simple info
        cv2.putText(output, f"Frame: {frame_num} | Cells: {cell_count}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_TEXT, 2)
        
        return output
    
    def toggle_trails(self):
        """Toggle trail visualization."""
        self.show_trails = not self.show_trails
        self.logger.info(f"Trails: {self.show_trails}")
    
    def toggle_ghosts(self):
        """Toggle ghost visualization."""
        self.show_ghosts = not self.show_ghosts
        self.logger.info(f"Ghosts: {self.show_ghosts}")
    
    def set_ghost_distance(self, frames_ahead: int):
        """Set how many frames ahead to predict ghosts."""
        self.ghost_frames_ahead = frames_ahead
        self.logger.info(f"Ghost prediction: {frames_ahead} frames ahead")
