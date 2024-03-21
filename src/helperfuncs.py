import cv2
import pygbif.species as gb

def resize_image(image, scale=35):
    scale_percent = scale  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    
    return resized_image

def find_names(search_images):
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

def compute_name_precision(image_names):
    """Compute correct name match rate"""
    matches = 0

    for tup in image_names:
        if tup[1] != "none":
            matches += 1
    
    match_rate = matches / len(image_names)

    return match_rate 