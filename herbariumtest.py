import requests
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from io import BytesIO

data_jpg = pd.read_csv("Herb.csv", usecols=[16])

# Extract the string from the list
image_url = data_jpg.values[0][0]

# Get the image data from the URL
response = requests.get('https://specify-attachments.science.ku.dk/static/NHMD_Botany/originals/' + image_url)

# Check if the request was successful
if response.status_code == 200:

    img = Image.open(BytesIO(response.content))

    # Convert image to greyscale and numpy array
    img_gray = img.convert('L')
    img_array = np.array(img_gray)

    plt.imshow(img_array, cmap="gray")
    plt.show()
else:
    print("Failed to retrieve the image.")