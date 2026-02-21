"""
Bio-Oracle - Frame-by-Frame Cell Counter
Uses traditional computer vision to detect ALL cells in each frame independently
No tracking - pure detection and counting per frame
"""
import cv2
import numpy as np
import argparse
import time
import logging
import csv
from pathlib import Path
from datetime import datetime
from src.core.traditional_detector import TraditionalCellDetector


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FrameByFrameCellCounter:
    """
    Count cells in each frame independently using traditional CV methods.
    """
    
    def __init__(self, min_area=400, max_area=25000, method='contours',
                 threshold_method='otsu', csv_output=None):
        """
        Initialize frame-by-frame cell counter.
        
        Args:
            min_area: Minimum cell area in pixels (default: 400)
            max_area: Maximum cell area in pixels (default: 25000 for large elongated cells)
            method: Detection method ('contours', 'blobs', or 'hybrid')
            threshold_method: Thresholding method ('adaptive', 'otsu', or 'simple')
            csv_output: Path to output CSV file
        """
        logger.info("Initializing Frame-by-Frame Cell Counter...")
        
        self.detector = TraditionalCellDetector(
            min_area=min_area,
            max_area=max_area,
            threshold_method=threshold_method
        )
        
        self.method = method
        self.csv_output = csv_output
        self.frame_data = []
        self.start_time = time.time()
        
        # Visualization settings
        self.colors = self._generate_colors(1000)
        self.bbox_thickness = 2
        self.text_scale = 0.6
        
        logger.info(f"Detection method: {method}")
        logger.info(f"Cell size range: {min_area}-{max_area} pixels²")
        logger.info("Framework initialized")
    
    def _generate_colors(self, num_colors: int):
        """Generate distinct colors."""
        np.random.seed(42)
        colors = []
        for i in range(num_colors):
            colors.append((
                int(np.random.randint(50, 255)),
                int(np.random.randint(50, 255)),
                int(np.random.randint(50, 255))
            ))
        return colors
    
    def process_frame(self, frame: np.ndarray, frame_num: int) -> tuple:
        """
        Process a single frame - detect and count all cells.
        
        Args:
            frame: Input frame
            frame_num: Frame number
            
        Returns:
            Tuple of (annotated_frame, cell_count, detections)
        """
        start_time = time.time()
        
        # Detect ALL cells in this frame
        detections = self.detector.detect(frame, method=self.method)
        cell_count = len(detections)
        
        # Record data
        self.frame_data.append({
            'frame': frame_num,
            'cell_count': cell_count,
            'timestamp': time.time(),
            'time_elapsed': time.time() - self.start_time,
            'processing_time': time.time() - start_time
        })
        
        # Visualize
        output_frame = frame.copy()
        
        # Draw each detection
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf = det
            color = self.colors[i % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(output_frame, 
                         (int(x1), int(y1)), (int(x2), int(y2)),
                         color, self.bbox_thickness)
            
            # Draw centroid
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(output_frame, (cx, cy), 3, color, -1)
            
            # Draw cell number
            label = f"{i+1}"
            cv2.putText(output_frame, label,
                       (int(x1) + 2, int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                       color, 1)
        
        # Draw info panel
        self._draw_info_panel(output_frame, cell_count, frame_num, time.time() - start_time)
        
        return output_frame, cell_count, detections
    
    def _draw_info_panel(self, frame: np.ndarray, cell_count: int, 
                         frame_num: int, processing_time: float):
        """Draw information panel."""
        h, w = frame.shape[:2]
        panel_height = 100
        panel_color = (40, 40, 40)
        
        # Semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), panel_color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Info text
        info_lines = [
            f"Bio-Oracle - Frame-by-Frame Cell Counter",
            f"Frame: {frame_num} | Cells Detected: {cell_count}",
            f"Method: {self.method.upper()} | Processing: {processing_time*1000:.1f}ms",
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 25 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Cell count prominently
        count_text = f"CELLS: {cell_count}"
        cv2.putText(frame, count_text, (w - 200, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    
    def calculate_per_second_averages(self, fps: float) -> list:
        """
        Calculate average cell count per second.
        
        Args:
            fps: Frames per second of the video
            
        Returns:
            List of dicts with per-second averages
        """
        if not self.frame_data:
            return []
        
        per_second = {}
        
        for frame_data in self.frame_data:
            frame_num = frame_data['frame']
            second = int(frame_num / fps)
            
            if second not in per_second:
                per_second[second] = []
            
            per_second[second].append(frame_data['cell_count'])
        
        # Calculate averages
        results = []
        for second in sorted(per_second.keys()):
            cell_counts = per_second[second]
            results.append({
                'second': second,
                'avg_cell_count': sum(cell_counts) / len(cell_counts),
                'min_cell_count': min(cell_counts),
                'max_cell_count': max(cell_counts),
                'frame_count': len(cell_counts)
            })
        
        return results
    
    def export_csv(self, fps: float = 25.0):
        """
        Export frame data and per-second averages to CSV.
        
        Args:
            fps: Frames per second of the video
        """
        if not self.csv_output or not self.frame_data:
            return
        
        csv_path = Path(self.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export per-frame data
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ['frame', 'cell_count', 'timestamp', 'time_elapsed', 'processing_time']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.frame_data)
        
        logger.info(f"Per-frame data exported to: {csv_path}")
        
        # Export per-second averages
        per_second_data = self.calculate_per_second_averages(fps)
        if per_second_data:
            per_second_path = csv_path.parent / f"{csv_path.stem}_per_second.csv"
            with open(per_second_path, 'w', newline='') as f:
                fieldnames = ['second', 'avg_cell_count', 'min_cell_count', 'max_cell_count', 'frame_count']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(per_second_data)
            
            logger.info(f"Per-second averages exported to: {per_second_path}")
    
    def print_summary(self, fps: float = 25.0):
        """
        Print summary statistics including per-second averages.
        
        Args:
            fps: Frames per second of the video
        """
        if not self.frame_data:
            return
        
        counts = [d['cell_count'] for d in self.frame_data]
        proc_times = [d['processing_time'] for d in self.frame_data]
        
        logger.info("=" * 70)
        logger.info("CELL COUNT SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total Frames: {len(self.frame_data)}")
        logger.info(f"Min Cells: {min(counts)}")
        logger.info(f"Max Cells: {max(counts)}")
        logger.info(f"Average Cells: {sum(counts)/len(counts):.2f}")
        logger.info(f"Total Cells Detected: {sum(counts)}")
        logger.info(f"Avg Processing Time: {sum(proc_times)/len(proc_times)*1000:.1f}ms/frame")
        logger.info("=" * 70)
        
        # Per-second summary
        per_second_data = self.calculate_per_second_averages(fps)
        if per_second_data:
            logger.info("")
            logger.info("PER-SECOND AVERAGES")
            logger.info("=" * 70)
            for data in per_second_data[:10]:  # Show first 10 seconds
                logger.info(f"Second {data['second']:3d}: Avg={data['avg_cell_count']:6.2f} "
                          f"(Min={data['min_cell_count']:3d}, Max={data['max_cell_count']:3d})")
            
            if len(per_second_data) > 10:
                logger.info(f"... ({len(per_second_data)-10} more seconds)")
            
            logger.info("=" * 70)
    
    def run_video(self, video_path: str, output_path: str = None,
                  display: bool = True, max_frames: int = None):
        """
        Run cell counting on video.
        
        Args:
            video_path: Path to video file (or 0 for webcam)
            output_path: Path to save output video
            display: Show video window
            max_frames: Maximum frames to process
        """
        logger.info(f"Processing video: {video_path}")
        
        fps = 25  # Default FPS
        
        # Open video
        if video_path == '0' or video_path == 0:
            cap = cv2.VideoCapture(0)
            logger.info("Using webcam")
        else:
            cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 25  # Default if FPS not available
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info(f"Saving to: {output_path}")
        
        frame_num = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                output_frame, cell_count, detections = self.process_frame(frame, frame_num)
                
                # Log progress
                if frame_num % 30 == 0:
                    logger.info(f"Frame {frame_num}/{total_frames}: {cell_count} cells")
                
                # Write output
                if writer:
                    writer.write(output_frame)
                
                # Display
                if display:
                    cv2.imshow('Cell Counter - Press Q to quit', output_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("User quit")
                        break
                    elif key == ord('d'):
                        # Show debug view
                        debug = self.detector.visualize_preprocessing(frame)
                        cv2.imshow('Debug View', debug)
                
                frame_num += 1
                
                if max_frames and frame_num >= max_frames:
                    break
        
        except KeyboardInterrupt:
            logger.info("Interrupted")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Export and summarize
            self.export_csv(fps)
            self.print_summary(fps)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Frame-by-Frame Cell Counter - Detects ALL cells in each frame'
    )
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file (or 0 for webcam)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to output video')
    parser.add_argument('--csv', type=str, default=None,
                       help='Path to CSV output')
    parser.add_argument('--method', type=str, default='contours',
                       choices=['edges', 'contours', 'blobs', 'hybrid'],
                       help='Detection method (contours = best for dark cells)')
    parser.add_argument('--threshold', type=str, default='otsu',
                       choices=['adaptive', 'otsu', 'simple'],
                       help='Thresholding method')
    parser.add_argument('--min-area', type=int, default=400,
                       help='Minimum cell area in pixels')
    parser.add_argument('--max-area', type=int, default=25000,
                       help='Maximum cell area in pixels')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process')
    
    args = parser.parse_args()
    
    # Auto-generate CSV path
    csv_path = args.csv
    if not csv_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(args.video).stem if args.video != '0' else 'webcam'
        csv_path = f"logs/cells_perframe_{video_name}_{timestamp}.csv"
    
    # Create counter
    counter = FrameByFrameCellCounter(
        min_area=args.min_area,
        max_area=args.max_area,
        method=args.method,
        threshold_method=args.threshold,
        csv_output=csv_path
    )
    
    # Run
    counter.run_video(
        video_path=args.video,
        output_path=args.output,
        display=not args.no_display,
        max_frames=args.max_frames
    )


if __name__ == '__main__':
    main()
