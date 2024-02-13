from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import easyocr
import cv2

from label_extractor import getNineLabels, runAgain

test_image_path = "exp5"

### LETS GET THE TXT ###
parent_directory = "runs"

#SÆT TIL TRUE HVIS DU VIL KØRE ALLE BILLEDER SÅ RUNS MAPPE BLIVER LAVET
#SÆT TIL FALSE HVIS DU HAR RUNS MAPPERNE
runAgain(doImages=False)

# FOLDER_PATH = KUN KØR FOR EN I STEDET FOR ALLE, F.EKS. folder_path=test_image_path
# Vil du køre alle billederne skal du bare ikke specificere folder_path
data_with_digit_9 = getNineLabels(parent_directory, folder_path=test_image_path)
print(data_with_digit_9)

for data in data_with_digit_9:
    image_path = 'herb_images/' + data[1]
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Check if the image is loaded properly
    if image is None:
        print(f"Failed to load image from {image_path}")
        continue

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_size = image_gray.shape[::-1]
    coordinates = data[0][1:5]
    print(coordinates)

    # YOLOv5-style coordinates: x, y, w, h
    x_center = float(coordinates[0]) * image_size[0]
    y_center = float(coordinates[1]) * image_size[1]
    box_width = float(coordinates[2]) * image_size[0]
    box_height = float(coordinates[3]) * image_size[1]

    # Calculate top-left and bottom-right coordinates of the bounding box
    x1 = int(x_center - (box_width / 2))
    y1 = int(y_center - (box_height / 2))
    x2 = int(x_center + (box_width / 2))
    y2 = int(y_center + (box_height / 2))

    # Draw bounding box on the image
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Resize the image
    scale_percent = 10  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Display the resized image with the bounding box using OpenCV
    cv2.imshow('Image with Bounding Box', resized_image)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()  # Close the OpenCV window after a key is pressed