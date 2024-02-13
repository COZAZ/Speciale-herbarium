from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import easyocr
import cv2

from label_extractor import getNineLabels, runAgain
from helperfuncs import resize_image

test_image_path = "exp104"

### LETS GET THE TXT ###
parent_directory = "runs"

#SÆT TIL TRUE HVIS DU VIL KØRE ALLE BILLEDER SÅ RUNS MAPPE BLIVER LAVET
#SÆT TIL FALSE HVIS DU HAR RUNS MAPPERNE
runAgain(doImages=False)

# FOLDER_PATH = KUN KØR FOR EN I STEDET FOR ALLE, F.EKS. folder_path=test_image_path
# Vil du køre alle billederne skal du bare ikke specificere folder_path
data_with_digit_9 = getNineLabels(parent_directory, folder_path=test_image_path)

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

    resized_image = resize_image(image, scale=10)

    # Display the resized image with the bounding box using OpenCV
    cv2.imshow('Image with Bounding Box', resized_image)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()  # Close the OpenCV window after a key is pressed

    # Crop the image
    cropped_image = image[y1:y2, x1:x2]

    # Perform OCR on the cropped image
    reader = easyocr.Reader(['en', 'da'])
    result = reader.readtext(cropped_image, detail=0)
    print(result)
    
    resized_cropped_image = resize_image(cropped_image, scale=35)

    # Show the cropped image
    cv2.imshow('Cropped Image', resized_cropped_image)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()  # Close the OpenCV window after a key is pressed