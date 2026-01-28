# Dataset Directory

This directory should contain your training images and labels.

## Required Structure

```
dataset/
├── images/
│   ├── train/          # Training images (.jpg, .png, etc.)
│   └── val/            # Validation images
└── labels/
    ├── train/          # Training labels (.txt files, YOLO format)
    └── val/            # Validation labels
```

## YOLO Label Format

Each image must have a corresponding `.txt` file with the same name:
- `cell_001.jpg` → `cell_001.txt`

Label file format (one line per object):
```
class_id center_x center_y width height
```

Where:
- `class_id`: Integer starting from 0
- `center_x, center_y`: Center of bounding box (normalized 0.0-1.0)
- `width, height`: Box dimensions (normalized 0.0-1.0)

Example:
```
0 0.5 0.5 0.1 0.15
0 0.3 0.7 0.08 0.12
```

## Labeling Tools

- **LabelImg**: https://github.com/heartexlabs/labelImg
- **CVAT**: https://github.com/opencv/cvat
- **Roboflow**: https://roboflow.com/

## Important

⚠️ **Do NOT commit this directory to Git!**
- Training images and labels can be very large
- Already added to `.gitignore`
