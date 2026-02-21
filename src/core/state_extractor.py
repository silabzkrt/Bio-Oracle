"""
Bio-Oracle State Extraction Module
Calculates population metrics from tracking data:
- Population Density (Cells/Area)
- Growth Rate (ΔCells/Δt)
- Velocity statistics
"""
import numpy as np
from typing import List, Dict, Tuple
import logging


class StateExtractor:
    """
    Extracts population-level statistics from cell tracking data.
    """
    
    def __init__(self, frame_width: int = 1280, frame_height: int = 720):
        """
        Initialize state extractor.
        
        Args:
            frame_width: Width of video frame in pixels
            frame_height: Height of video frame in pixels
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_area = frame_width * frame_height
        self.logger = logging.getLogger(__name__)
        
        # History for calculating rates
        self.cell_counts_history = []
        self.timestamps_history = []
        self.density_history = []
        
    def calculate_population_density(self, cell_count: int) -> float:
        """
        Calculate population density: ρ = Cells / Area
        
        Args:
            cell_count: Number of cells detected
            
        Returns:
            Population density (cells per pixel)
        """
        density = cell_count / self.frame_area
        self.density_history.append(density)
        return density
    
    def calculate_growth_rate(self, current_count: int, timestamp: float) -> float:
        """
        Calculate growth rate: r = ΔCells / Δt
        
        Args:
            current_count: Current number of cells
            timestamp: Current timestamp in seconds
            
        Returns:
            Growth rate (cells per second)
        """
        self.cell_counts_history.append(current_count)
        self.timestamps_history.append(timestamp)
        
        # Need at least 2 data points
        if len(self.cell_counts_history) < 2:
            return 0.0
        
        # Calculate rate from recent history (last 5 samples)
        window_size = min(5, len(self.cell_counts_history))
        recent_counts = self.cell_counts_history[-window_size:]
        recent_times = self.timestamps_history[-window_size:]
        
        # Linear regression for smoothed rate
        delta_cells = recent_counts[-1] - recent_counts[0]
        delta_time = recent_times[-1] - recent_times[0]
        
        if delta_time == 0:
            return 0.0
        
        growth_rate = delta_cells / delta_time
        return growth_rate
    
    def calculate_per_capita_growth_rate(self, current_count: int, timestamp: float) -> float:
        """
        Calculate per-capita growth rate: r/N = (1/N)(dN/dt)
        
        Args:
            current_count: Current number of cells
            timestamp: Current timestamp
            
        Returns:
            Per-capita growth rate
        """
        growth_rate = self.calculate_growth_rate(current_count, timestamp)
        
        if current_count == 0:
            return 0.0
        
        return growth_rate / current_count
    
    def calculate_average_velocity(self, entities: List) -> Tuple[float, float]:
        """
        Calculate average velocity of all tracked entities.
        
        Args:
            entities: List of BioEntity objects
            
        Returns:
            Tuple of (average_speed, average_direction_degrees)
        """
        if not entities:
            return 0.0, 0.0
        
        speeds = []
        directions = []
        
        for entity in entities:
            if entity.velocity is not None:
                vx, vy = entity.velocity
                speed = np.sqrt(vx**2 + vy**2)
                direction = np.arctan2(vy, vx) * 180 / np.pi
                
                speeds.append(speed)
                directions.append(direction)
        
        if not speeds:
            return 0.0, 0.0
        
        avg_speed = np.mean(speeds)
        avg_direction = np.mean(directions)
        
        return avg_speed, avg_direction
    
    def get_population_state(self, cell_count: int, timestamp: float, 
                           entities: List = None) -> Dict:
        """
        Get complete population state.
        
        Args:
            cell_count: Current cell count
            timestamp: Current timestamp
            entities: List of tracked entities (optional)
            
        Returns:
            Dictionary with population metrics
        """
        density = self.calculate_population_density(cell_count)
        growth_rate = self.calculate_growth_rate(cell_count, timestamp)
        per_capita_rate = self.calculate_per_capita_growth_rate(cell_count, timestamp)
        
        state = {
            'cell_count': cell_count,
            'timestamp': timestamp,
            'density': density,
            'density_per_1000px': density * 1000,  # More readable unit
            'growth_rate': growth_rate,  # cells/second
            'per_capita_growth_rate': per_capita_rate,  # 1/second
        }
        
        # Add velocity statistics if entities provided
        if entities:
            avg_speed, avg_direction = self.calculate_average_velocity(entities)
            state['avg_speed'] = avg_speed
            state['avg_direction'] = avg_direction
        
        return state
    
    def get_recent_trend(self, window_seconds: float = 5.0) -> str:
        """
        Get population trend over recent window.
        
        Args:
            window_seconds: Time window to analyze
            
        Returns:
            Trend description ('growing', 'declining', 'stable')
        """
        if len(self.cell_counts_history) < 10:
            return 'insufficient_data'
        
        # Look at recent window
        recent_times = []
        recent_counts = []
        
        current_time = self.timestamps_history[-1]
        for i in range(len(self.timestamps_history) - 1, -1, -1):
            if current_time - self.timestamps_history[i] <= window_seconds:
                recent_times.append(self.timestamps_history[i])
                recent_counts.append(self.cell_counts_history[i])
            else:
                break
        
        if len(recent_counts) < 2:
            return 'insufficient_data'
        
        # Calculate trend
        start_avg = np.mean(recent_counts[:len(recent_counts)//2])
        end_avg = np.mean(recent_counts[len(recent_counts)//2:])
        
        change_percent = ((end_avg - start_avg) / start_avg * 100) if start_avg > 0 else 0
        
        if change_percent > 5:
            return 'growing'
        elif change_percent < -5:
            return 'declining'
        else:
            return 'stable'
    
    def reset(self):
        """Reset all history."""
        self.cell_counts_history = []
        self.timestamps_history = []
        self.density_history = []
