"""
Test module for find_cell.py using EMDS5-Original dataset
Tests cell detection, segmentation, and coordinate extraction on microscopy images
"""
import find_cell
import cv2 as opencv
import os


def test_all_images():
    data_folder = "EMDS5-Original"
    
    # Get all PNG images from the folder
    image_files = [f for f in os.listdir(data_folder) if f.endswith('.png')]
    
    print(f"Testing {len(image_files)} images from {data_folder}\n")
    
    success_count = 0
    failure_count = 0
    
    for image_file in image_files:
        image_path = os.path.join(data_folder, image_file)
        print(f"Processing: {image_file}")
        
        # Test find_cell function
        cell_image = find_cell.find_cell(image_path)
        
        # Test return_coordinates function
        coordinates = find_cell.return_coordinates(image_path)
        
        # Test segmentation function
        segmented_image = find_cell.segmentation(image_path)
        
        if cell_image is not None and coordinates is not None and segmented_image is not None:
            x, y, w, h = coordinates
            print(f"  ✓ Cell detected at ({x}, {y}) with size {w}x{h}")
            success_count += 1
        else:
            print(f"  ✗ No cell detected")
            failure_count += 1
    
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"  Successful: {success_count}/{len(image_files)}")
    print(f"  Failed: {failure_count}/{len(image_files)}")
    print(f"  Success Rate: {(success_count/len(image_files)*100):.1f}%")


def test_single_image(image_name):
    """Test a single image with visualization"""
    data_folder = "EMDS5-Original"
    image_path = os.path.join(data_folder, image_name)
    
    if not os.path.exists(image_path):
        print(f"Error: Image {image_name} not found in {data_folder}")
        return
    
    print(f"Testing: {image_name}")
    
    # Test and display cell detection
    cell_image = find_cell.find_cell(image_path)
    if cell_image is not None:
        opencv.imshow("Detected Cell", cell_image)
    
    # Test and display segmentation with green rectangle
    segmented_image = find_cell.segmentation(image_path)
    if segmented_image is not None:
        opencv.imshow("Segmentation - Green Rectangle", segmented_image)
        
    # Display coordinates
    coordinates = find_cell.return_coordinates(image_path)
    if coordinates is not None:
        x, y, w, h = coordinates
        print(f"Cell coordinates: ({x}, {y}), Size: {w}x{h}")
    
    opencv.waitKey(0)
    opencv.destroyAllWindows()


def test_sample_images():
    """Test a sample of images from different groups"""
    sample_images = [
        "EMDS5-g01-01.png",
        "EMDS5-g05-10.png",
        "EMDS5-g10-15.png",
        "EMDS5-g15-20.png",
        "EMDS5-g21-05.png"
    ]
    
    data_folder = "EMDS5-Original"
    
    for image_name in sample_images:
        image_path = os.path.join(data_folder, image_name)
        
        if os.path.exists(image_path):
            print(f"\n{image_name}:")
            coordinates = find_cell.return_coordinates(image_path)
            cell_image = find_cell.find_cell(image_path)
            
            if cell_image is not None and coordinates is not None:
                x, y, w, h = coordinates
                print(f"  Cell found at ({x}, {y}), Size: {w}x{h}")
            else:
                print(f"  No cell detected")


if __name__ == "__main__":
    print("EMDS5-Original Dataset Test Module")
    print("="*50)
    print("\nSelect test mode:")
    print("1. Test all images (no visualization)")
    print("2. Test sample images (console output)")
    print("3. Test single image with visualization")
    
    choice = input("\nEnter choice (1-3): ")
    
    if choice == "1":
        test_all_images()
    elif choice == "2":
        test_sample_images()
    elif choice == "3":
        image_name = input("Enter image filename (e.g., EMDS5-g01-01.png): ")
        test_single_image(image_name)
    else:
        print("Invalid choice. Running sample test...")
        test_sample_images()