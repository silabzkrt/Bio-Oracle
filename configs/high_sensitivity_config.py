"""
High Sensitivity Cell Detection Configuration
Optimized to detect EVERY cell in each frame
"""

# Very low confidence threshold to catch all possible cells
CONFIDENCE_THRESHOLD = 0.25  # Lower = more detections

# Lower IoU threshold to allow overlapping detections
IOU_THRESHOLD = 0.3  # Lower = more overlapping boxes allowed

# Tracker settings for dense cell populations
MAX_DISTANCE = 80.0  # Allow larger matching distance
MAX_DISAPPEARED = 50  # Keep tracking longer

# Detection settings
DETECT_ALL_CLASSES = True  # Don't filter by class
MIN_BBOX_AREA = 50  # Smaller minimum size to catch tiny cells

print(f"""
High Sensitivity Detection Mode:
- Confidence: {CONFIDENCE_THRESHOLD} (Very Low - Detects More)
- IoU: {IOU_THRESHOLD} (Allows Overlaps)
- Max Distance: {MAX_DISTANCE}px (Tracks Fast Movement)
- Min Size: {MIN_BBOX_AREA}px² (Catches Small Cells)
""")
