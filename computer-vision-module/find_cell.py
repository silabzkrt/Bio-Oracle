import cv2 as opencv


def find_cell(image_path):
    image = opencv.imread(image_path)
    gray = opencv.cvtColor(image, opencv.COLOR_BGR2GRAY)
    blurred = opencv.GaussianBlur(gray, (5, 5), 0)
    edges = opencv.Canny(blurred, 50, 150)
    contours, _ = opencv.findContours(edges, opencv.RETR_EXTERNAL,
                                      opencv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    largest_contour = max(contours, key=opencv.contourArea)
    x, y, w, h = opencv.boundingRect(largest_contour)
    return True


def return_coordinates(image_path):
    image = opencv.imread(image_path)    
    gray = opencv.cvtColor(image, opencv.COLOR_BGR2GRAY)    
    blurred = opencv.GaussianBlur(gray, (5, 5), 0)
    edges = opencv.Canny(blurred, 50, 150)
    contours, _ = opencv.findContours(edges, opencv.RETR_EXTERNAL, 
                                      opencv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=opencv.contourArea)
    x, y, w, h = opencv.boundingRect(largest_contour)
    return (x, y, w, h)


def segmentation(image_path):
    # Draw a green rectangle that covers the cell
    image = opencv.imread(image_path)
    coordinates = return_coordinates(image_path)
    if coordinates is None:
        return None
    x, y, w, h = coordinates
    opencv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image


if __name__ == "__main__":
    image_path = "path_to_your_image.jpg"
    cell_image = find_cell(image_path)
    if cell_image is not None:
        opencv.imshow("Detected Cell", cell_image)
        opencv.waitKey(0)
    else:
        print("No cell detected.")


