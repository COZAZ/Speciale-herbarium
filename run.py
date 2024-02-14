from label_extractor import getInstituteLabels, predictLabels
from helperfuncs import resize_image
import easyocr
import cv2
import pygbif.species as gb

def load_and_preprocess_image(image_path):
    """Load and preprocess the image."""
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load image from {image_path}")
        return None, None

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_size = image_gray.shape[::-1]

    return image, image_size

def calculate_coordinates(image_size, coordinates):
    """Calculate bounding box coordinates."""
    x_center, y_center, box_width, box_height = map(float, coordinates)
    x_center *= image_size[0]
    y_center *= image_size[1]
    box_width *= image_size[0]
    box_height *= image_size[1]

    x1, y1 = int(x_center - (box_width / 2)), int(y_center - (box_height / 2))
    x2, y2 = int(x_center + (box_width / 2)), int(y_center + (box_height / 2))

    return x1, y1, x2, y2

def display_image_with_bbox(image, bbox, scale=10):
    """Display image with bounding box."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    resized_image = resize_image(image, scale=scale)
    cv2.imshow('Image with Bounding Box', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_cropped_image(image, bbox):
    """Process cropped image."""
    cropped_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    reader = easyocr.Reader(['en', 'da', 'la', 'de'], gpu=False)
    result = reader.readtext(cropped_image, detail=0, paragraph=True)

    # UNCOMMENT IF YOU WANT TO SEE THE CROPPED RESULT
    #resized_cropped_image = resize_image(cropped_image, scale=35)
    #cv2.imshow('Cropped Image', resized_cropped_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return result

def process_image_data(institute_label_data):
    """Process image data."""

    ocr_results = []

    for data in institute_label_data:
        image_path = 'herb_images/' + data[1]
        image, image_size = load_and_preprocess_image(image_path)

        if image is None:
            continue

        coordinates = data[0][1:5]
        bbox = calculate_coordinates(image_size, coordinates)

        # UNCOMMENT IF YOU WANT TO SEE THE DRAWN BOUNDING BOX
        #display_image_with_bbox(image, bbox)

        text = process_cropped_image(image, bbox)
        processed_image_info = (data[1], text)
        
        ocr_results.append(processed_image_info)
    
    return ocr_results

def findNames(search_images):
    """Identify the real plant names of the OCR results"""

    identified_image_names = []

    for ocr_res in search_images:
        image_name = ocr_res[0]
        image_text = ocr_res[1]

        best_match = "none"
        best_score = 0
        for text_elm in image_text:
            lookup = gb.name_backbone(name=text_elm, kingdom="plants")

            if 'scientificName' in lookup:
  
                current_conf = lookup['confidence']
                current_name = lookup['scientificName']

                if current_conf >= best_score:
                    best_score = current_conf
                    best_match = current_name
        
        identified_image_names.append((image_name, best_match, best_score))
    
    return identified_image_names

def main():
    parent_directory = "runs"
    test_image_path = "exp"

    ### PIPELINE step 1: Identify bounding boxes ###
    # Set doImages to True to predict labels of all images in herb_images
    # Set to false to skip this step, if you already have the "runs" results
    predictLabels(doImages=False)

    ### PIPELINE step 2: Find label location in images ###
    # Set folder_path to specific exp folder to get label location of only one image
    # To run on all images, do not set the folder_path parameter
    institute_label_data = getInstituteLabels(parent_directory)

    ### PIPELINE step 3: Extract text from images ###
    # Performs OCR on cropped images according to the predicted bounding box locations
    processed_images = process_image_data(institute_label_data)

    ### LÆS: GBIF TEST STUFF BELOW ###

    found_plant_names = findNames(processed_images)

    print(found_plant_names)

if __name__ == "__main__":
    main()