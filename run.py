import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import easyocr
import tensorflow_hub as hub

from tensorflowhelper import run_detector

image_path = 'herb_images/largetest.jpg'

test_image = Image.open(image_path) # Update path to your own data set location
test_image = test_image.convert('L')
# Define the coordinates of the region you want to zoom in on
# (x1, y1) is the top-left corner, and (x2, y2) is the bottom-right corner of the region
x1, y1 = 4000, 7000  # example coordinates
x2, y2 = 6500, 9200  # example coordinates
"""
# Crop the image to the specified region
cropped_image = test_image.crop((x1, y1, x2, y2))
cropped_image = np.array(cropped_image)
print(type(cropped_image))

plt.imshow(cropped_image, cmap="gray")
plt.show()

reader = easyocr.Reader(['en', 'da'])
result = reader.readtext(cropped_image)
print(result)

# Create a Matplotlib figure and axis
plt.figure(figsize=(10, 10))
plt.imshow(cropped_image, cmap='gray')

# Plot bounding boxes on the image
for detection in result:
    # Extract the bounding box coordinates
    box = detection[0]  # The bounding box information is usually at index 1
    x1, y1 = box[0]
    x2, y2 = box[1]
    x3, y3 = box[2]
    x4, y4 = box[3]

    plt.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], 'r', linewidth=2)
# Show the image with bounding boxes
plt.axis('off')
plt.show()
"""
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"

detector = hub.load(module_handle).signatures['default']

run_detector(detector, image_path)