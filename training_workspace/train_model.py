"""
YOLO11 Cell Detection Model Training Script
Use this script to train your custom cell detection model
"""

from ultralytics import YOLO
import os


def train_cell_detector(
    data_yaml='data.yaml',
    epochs=100,
    imgsz=640,
    batch_size=16,
    model_name='yolo11n.pt',
    device='cpu',
    project='runs/train',
    name='cell_detector'
):
    """
    Train a YOLO11 model for cell detection
    
    Args:
        data_yaml (str): Path to data.yaml configuration file
        epochs (int): Number of training epochs
        imgsz (int): Input image size
        batch_size (int): Batch size for training
        model_name (str): Pretrained model to start from
        device (str): Device to train on ('cpu', 'cuda', '0', etc.)
        project (str): Project directory for saving results
        name (str): Name for this training run
    """
    
    # Initialize model (starts with pretrained weights)
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)
    
    # Train the model
    print("Starting training...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        
        # Optimization settings
        patience=50,  # Early stopping patience
        save=True,    # Save checkpoints
        
        # Augmentation settings
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,    # HSV-Saturation augmentation
        hsv_v=0.4,    # HSV-Value augmentation
        degrees=0.0,  # Rotation augmentation
        translate=0.1, # Translation augmentation
        scale=0.5,    # Scale augmentation
        flipud=0.0,   # Vertical flip probability
        fliplr=0.5,   # Horizontal flip probability
        mosaic=1.0,   # Mosaic augmentation probability
    )
    
    print("\nTraining complete!")
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")
    
    # Validate the model
    print("\nValidating model...")
    metrics = model.val()
    
    print("\nValidation Results:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    return model, results


def main():
    """Main training function"""
    
    # Check if data.yaml exists
    if not os.path.exists('data.yaml'):
        print("ERROR: data.yaml not found!")
        print("Please create a data.yaml file with your dataset configuration.")
        print("\nExample data.yaml:")
        print("""
# Path to dataset root
path: ./dataset

# Paths to train/val/test splits (relative to 'path')
train: images/train
val: images/val
test: images/test  # optional

# Class names
names:
  0: cell
  1: moving_cell
  2: staying_cell
        """)
        return
    
    # Training configuration
    CONFIG = {
        'data_yaml': 'data.yaml',
        'epochs': 100,           # Adjust based on your dataset size
        'imgsz': 640,            # Image size
        'batch_size': 16,        # Adjust based on your GPU memory
        'model_name': 'yolo11n.pt',  # n=nano, s=small, m=medium, l=large, x=extra-large
        'device': 'cpu',         # Change to '0' for GPU or 'cuda'
        'project': 'runs/train',
        'name': 'cell_detector'
    }
    
    print("Training Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("\n")
    
    # Start training
    model, results = train_cell_detector(**CONFIG)
    
    print("\n" + "="*60)
    print("Training finished successfully!")
    print("="*60)
    print(f"\nTo use your trained model:")
    print(f"1. Copy the best.pt file to: ../assets/models/best.pt")
    print(f"2. Update config.py if needed")
    print(f"3. Run: python main.py")


if __name__ == '__main__':
    main()
