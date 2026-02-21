"""
Bio-Oracle Framework - Unified Main Application
Combines computer vision, cell tracking, and population dynamics prediction
Day 1 + Day 2 integrated implementation
"""
import cv2
import yaml
import time
import argparse
import logging
from pathlib import Path

from core import VisionManager, CentroidTracker
from simulation import StateExtractor, PopulationOracle
from utils import Visualizer, DataLogger


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BioOracle:
    """
    Main Bio-Oracle application combining:
    - Computer Vision (YOLO or Traditional CV)
    - Cell Tracking (Centroid-based)
    - State Extraction (Population metrics)
    - Population Prediction (Lotka-Volterra)
    - Visualization (Trails, ghosts, metrics)
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize Bio-Oracle from configuration file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        logger.info("Initializing Bio-Oracle Framework...")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self._init_vision()
        self._init_tracker()
        self._init_simulation()
        self._init_visualization()
        self._init_data_logging()
        
        # Runtime state
        self.start_time = time.time()
        self.frame_count = 0
        self.paused = False
        
        logger.info("Bio-Oracle initialized successfully")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            config = self._default_config()
        
        return config
    
    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            'vision': {
                'detection_method': 'traditional',
                'model_path': 'models/yolov11n.pt',
                'confidence_threshold': 0.5,
                'device': 'cpu',
                'min_area': 400,
                'max_area': 25000
            },
            'tracking': {
                'max_disappeared': 30,
                'max_distance': 75.0
            },
            'simulation': {
                'growth_rate': 0.1,
                'carrying_capacity': 200.0,
                'prediction_seconds': 10.0
            },
            'visualization': {
                'show_trails': True,
                'show_ghosts': True,
                'ghost_frames_ahead': 10
            },
            'logging': {
                'enabled': True,
                'output_dir': 'data/outputs'
            }
        }
    
    def _init_vision(self):
        """Initialize vision manager."""
        vision_cfg = self.config['vision']
        self.vision = VisionManager(
            detection_method=vision_cfg.get('detection_method', 'traditional'),
            model_path=vision_cfg.get('model_path', 'models/yolov11n.pt'),
            conf_threshold=vision_cfg.get('confidence_threshold', 0.5),
            device=vision_cfg.get('device', 'cpu'),
            min_area=vision_cfg.get('min_area', 400),
            max_area=vision_cfg.get('max_area', 25000)
        )
        logger.info(f"Vision: {vision_cfg['detection_method']}")
    
    def _init_tracker(self):
        """Initialize centroid tracker."""
        track_cfg = self.config['tracking']
        self.tracker = CentroidTracker(
            max_disappeared=track_cfg.get('max_disappeared', 30),
            max_distance=track_cfg.get('max_distance', 75.0)
        )
        logger.info(f"Tracker: max_disappeared={track_cfg['max_disappeared']}")
    
    def _init_simulation(self):
        """Initialize state extractor and oracle."""
        sim_cfg = self.config['simulation']
        
        self.state_extractor = None  # Initialized with video dimensions
        
        self.oracle = PopulationOracle(
            initial_population=10.0,
            growth_rate=sim_cfg.get('growth_rate', 0.1),
            carrying_capacity=sim_cfg.get('carrying_capacity', 200.0)
        )
        
        self.prediction_seconds = sim_cfg.get('prediction_seconds', 10.0)
        logger.info(f"Oracle: K={sim_cfg['carrying_capacity']}, r={sim_cfg['growth_rate']}")
    
    def _init_visualization(self):
        """Initialize visualizer."""
        vis_cfg = self.config['visualization']
        self.visualizer = Visualizer(
            show_trails=vis_cfg.get('show_trails', True),
            show_ghosts=vis_cfg.get('show_ghosts', True),
            ghost_frames_ahead=vis_cfg.get('ghost_frames_ahead', 10)
        )
        logger.info("Visualizer: Ready")
    
    def _init_data_logging(self):
        """Initialize data logger."""
        log_cfg = self.config['logging']
        
        if log_cfg.get('enabled', True):
            self.data_logger = DataLogger(
                output_dir=log_cfg.get('output_dir', 'data/outputs')
            )
            logger.info(f"DataLogger: {self.data_logger.session_name}")
        else:
            self.data_logger = None
    
    def process_frame(self, frame, frame_num: int):
        """
        Process a single frame through the complete pipeline.
        
        Args:
            frame: Input BGR frame
            frame_num: Frame number
            
        Returns:
            Tuple of (output_frame, cell_count, population_state, prediction_text, processing_time)
        """
        start_time = time.time()
        
        # Initialize state extractor with frame dimensions (first frame only)
        if self.state_extractor is None:
            h, w = frame.shape[:2]
            self.state_extractor = StateExtractor(w, h)
            logger.info(f"Frame dimensions: {w}x{h}")
        
        # 1. Detect cells
        detections = self.vision.detect(frame)
        
        # 2. Update tracker
        entities = self.tracker.update(detections)
        
        # 3. Extract population state
        current_time = time.time() - self.start_time
        population_state = self.state_extractor.get_population_state(
            cell_count=len(entities),
            timestamp=current_time,
            entities=list(entities.values())
        )
        
        # 4. Update oracle and get prediction
        self.oracle.update(len(entities), current_time)
        prediction_text = self.oracle.get_prediction_text(self.prediction_seconds)
        
        # 5. Visualize
        output_frame = self.visualizer.visualize_frame(
            frame, entities, population_state, prediction_text, frame_num
        )
        
        # 6. Log data
        if self.data_logger:
            self.data_logger.log_frame(frame_num, len(entities), population_state, current_time)
            self.data_logger.log_prediction(frame_num, prediction_text)
        
        processing_time = time.time() - start_time
        
        return output_frame, len(entities), population_state, prediction_text, processing_time
    
    def run_video(self, video_path: str, output_path: str = None, 
                 display: bool = True, save_data: bool = True):
        """
        Run Bio-Oracle on a video file.
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save output video
            display: Whether to display video in window
            save_data: Whether to save CSV/JSON logs
        """
        logger.info(f"Processing: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
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
        total_time = 0
        
        try:
            while True:
                if not self.paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    output_frame, cell_count, state, pred_text, proc_time = self.process_frame(
                        frame, frame_num
                    )
                    
                    total_time += proc_time
                    
                    # Log progress
                    if frame_num % 30 == 0:
                        avg_fps = 1.0 / (total_time / max(frame_num, 1))
                        logger.info(f"Frame {frame_num}/{total_frames}: {cell_count} cells, "
                                  f"{pred_text}, {avg_fps:.1f} FPS")
                    
                    # Write output
                    if writer:
                        writer.write(output_frame)
                    
                    frame_num += 1
                else:
                    output_frame = frame  # Use last frame when paused
                
                # Display
                if display:
                    cv2.imshow('Bio-Oracle - Q:Quit | G:Ghosts | T:Trails | Space:Pause', output_frame)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        logger.info("User quit")
                        break
                    elif key == ord('g'):
                        self.visualizer.toggle_ghosts()
                    elif key == ord('t'):
                        self.visualizer.toggle_trails()
                    elif key == ord(' '):
                        self.paused = not self.paused
                        logger.info(f"Paused: {self.paused}")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            
            # Save data logs
            if save_data and self.data_logger:
                self.data_logger.export_csv()
                self.data_logger.export_summary()
                logger.info("Data exported successfully")
            
            logger.info(f"Processing complete: {frame_num} frames")
            
            # Print final statistics
            if self.data_logger:
                stats = self.data_logger.get_statistics()
                logger.info(f"Average cells: {stats.get('avg_cells', 0):.1f}")
    
    def run_webcam(self, camera_index: int = 0):
        """
        Run Bio-Oracle on webcam feed.
        
        Args:
            camera_index: Camera device index (usually 0)
        """
        logger.info(f"Opening webcam (index {camera_index})...")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error("Failed to open webcam")
            return
        
        frame_num = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from webcam")
                    break
                
                # Process frame
                output_frame, cell_count, state, pred_text, proc_time = self.process_frame(
                    frame, frame_num
                )
                
                # Display
                cv2.imshow('Bio-Oracle Webcam - Q:Quit | G:Ghosts | T:Trails', output_frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('g'):
                    self.visualizer.toggle_ghosts()
                elif key == ord('t'):
                    self.visualizer.toggle_trails()
                
                frame_num += 1
        
        except KeyboardInterrupt:
            logger.info("Interrupted")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Webcam session ended")


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(
        description='Bio-Oracle: Cell Tracking & Population Prediction Framework'
    )
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--webcam', type=int, nargs='?', const=0, 
                       help='Use webcam (optionally specify camera index)')
    parser.add_argument('--output', type=str, help='Path to save output video')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display (faster processing)')
    parser.add_argument('--no-save', action='store_true',
                       help='Disable data logging')
    
    args = parser.parse_args()
    
    # Create Bio-Oracle instance
    oracle = BioOracle(config_path=args.config)
    
    # Run on video or webcam
    if args.video:
        oracle.run_video(
            video_path=args.video,
            output_path=args.output,
            display=not args.no_display,
            save_data=not args.no_save
        )
    elif args.webcam is not None:
        oracle.run_webcam(camera_index=args.webcam)
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python main.py --video data/input_samples/video.mp4")
        print("  python main.py --video data/input_samples/video.mp4 --output data/outputs/result.mp4")
        print("  python main.py --webcam")


if __name__ == '__main__':
    main()
