"""
DataLogger - Handles CSV export and data persistence for Bio-Oracle
Logs population metrics, cell counts, and tracking data over time
"""
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import logging


class DataLogger:
    """
    Handles logging of population data and tracking metrics to CSV and JSON.
    """
    
    def __init__(self, output_dir: str = "data/outputs", 
                 session_name: str = None):
        """
        Initialize data logger.
        
        Args:
            output_dir: Directory to save log files
            session_name: Optional session name (auto-generated if None)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if session_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"biooracle_{timestamp}"
        
        self.session_name = session_name
        self.logger = logging.getLogger(__name__)
        
        # Data buffers
        self.frame_data = []
        self.population_data = []
        
        self.logger.info(f"DataLogger initialized: {session_name}")
    
    def log_frame(self, frame_num: int, cell_count: int, 
                 population_state: Dict = None, timestamp: float = None):
        """
        Log data for a single frame.
        
        Args:
            frame_num: Frame number
            cell_count: Number of cells detected
            population_state: Optional population metrics
            timestamp: Optional timestamp
        """
        record = {
            'frame': frame_num,
            'cell_count': cell_count,
            'timestamp': timestamp if timestamp is not None else frame_num
        }
        
        # Add population metrics if available
        if population_state:
            record.update({
                'density': population_state.get('density', 0),
                'density_per_1000px': population_state.get('density_per_1000px', 0),
                'growth_rate': population_state.get('growth_rate', 0),
                'per_capita_growth_rate': population_state.get('per_capita_growth_rate', 0),
                'avg_speed': population_state.get('avg_speed', 0),
                'avg_direction': population_state.get('avg_direction', 0)
            })
        
        self.frame_data.append(record)
    
    def log_prediction(self, frame_num: int, prediction_text: str, 
                      predicted_population: float = None):
        """
        Log Oracle prediction.
        
        Args:
            frame_num: Frame number when prediction was made
            prediction_text: Prediction description
            predicted_population: Predicted population value
        """
        record = {
            'frame': frame_num,
            'prediction_text': prediction_text,
            'predicted_population': predicted_population
        }
        
        self.population_data.append(record)
    
    def export_csv(self, filename: str = None) -> Path:
        """
        Export frame data to CSV.
        
        Args:
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to saved CSV file
        """
        if filename is None:
            filename = f"{self.session_name}_frames.csv"
        
        csv_path = self.output_dir / filename
        
        if not self.frame_data:
            self.logger.warning("No data to export")
            return csv_path
        
        # Get all keys from first record
        fieldnames = list(self.frame_data[0].keys())
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.frame_data)
        
        self.logger.info(f"Exported {len(self.frame_data)} records to {csv_path}")
        return csv_path
    
    def export_json(self, filename: str = None) -> Path:
        """
        Export all data to JSON.
        
        Args:
            filename: Optional filename
            
        Returns:
            Path to saved JSON file
        """
        if filename is None:
            filename = f"{self.session_name}_data.json"
        
        json_path = self.output_dir / filename
        
        data = {
            'session_name': self.session_name,
            'timestamp': datetime.now().isoformat(),
            'frame_count': len(self.frame_data),
            'frame_data': self.frame_data,
            'population_data': self.population_data
        }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Exported session data to {json_path}")
        return json_path
    
    def export_summary(self, filename: str = None) -> Path:
        """
        Export statistical summary to text file.
        
        Args:
            filename: Optional filename
            
        Returns:
            Path to saved summary file
        """
        if filename is None:
            filename = f"{self.session_name}_summary.txt"
        
        summary_path = self.output_dir / filename
        
        if not self.frame_data:
            self.logger.warning("No data for summary")
            return summary_path
        
        # Calculate statistics
        cell_counts = [d['cell_count'] for d in self.frame_data]
        
        stats = {
            'Total Frames': len(self.frame_data),
            'Average Cells': sum(cell_counts) / len(cell_counts),
            'Min Cells': min(cell_counts),
            'Max Cells': max(cell_counts),
            'Total Cells Detected': sum(cell_counts)
        }
        
        # Write summary
        with open(summary_path, 'w') as f:
            f.write(f"Bio-Oracle Session Summary\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Session: {self.session_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for key, value in stats.items():
                f.write(f"{key}: {value:.2f if isinstance(value, float) else value}\n")
            
            # Add growth rate info if available
            if self.frame_data and 'growth_rate' in self.frame_data[0]:
                growth_rates = [d.get('growth_rate', 0) for d in self.frame_data 
                              if 'growth_rate' in d]
                if growth_rates:
                    avg_growth = sum(growth_rates) / len(growth_rates)
                    f.write(f"\nAverage Growth Rate: {avg_growth:+.4f} cells/s\n")
        
        self.logger.info(f"Exported summary to {summary_path}")
        return summary_path
    
    def get_statistics(self) -> Dict:
        """
        Get current statistics dictionary.
        
        Returns:
            Dictionary with session statistics
        """
        if not self.frame_data:
            return {}
        
        cell_counts = [d['cell_count'] for d in self.frame_data]
        
        return {
            'session_name': self.session_name,
            'frame_count': len(self.frame_data),
            'avg_cells': sum(cell_counts) / len(cell_counts),
            'min_cells': min(cell_counts),
            'max_cells': max(cell_counts)
        }
    
    def clear(self):
        """Clear all logged data."""
        self.frame_data = []
        self.population_data = []
        self.logger.info("Data logger cleared")
