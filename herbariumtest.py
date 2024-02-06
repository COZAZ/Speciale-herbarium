import requests
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from io import BytesIO

all_data = pd.read_csv("Herb.csv")
data_jpg = pd.read_csv("Herb.csv", usecols=[16])

print(data_jpg.values[0])

# Extract the string from the list
image_url = data_jpg.values[0][0]

# Get the image data from the URL
response = requests.get('https://specify-attachments.science.ku.dk/static/NHMD_Botany/originals/' + image_url)

# Check if the request was successful
if response.status_code == 200:
    # Open and display the image using PIL
    img = Image.open(BytesIO(response.content))
    img.show()
else:
    print("Failed to retrieve the image.")