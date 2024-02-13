from label_extractor import getNineLabels, runAgain
from helperfuncs import resize_image
import easyocr
import cv2

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
    result = reader.readtext(cropped_image, detail=0)
    print(result)

    resized_cropped_image = resize_image(cropped_image, scale=35)
    cv2.imshow('Cropped Image', resized_cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_image_data(data_with_digit_9):
    """Process image data."""
    for data in data_with_digit_9:
        image_path = 'herb_images/' + data[1]
        image, image_size = load_and_preprocess_image(image_path)

        if image is None:
            continue

        coordinates = data[0][1:5]
        bbox = calculate_coordinates(image_size, coordinates)

        display_image_with_bbox(image, bbox)

        process_cropped_image(image, bbox)

def main():
    parent_directory = "runs"
    test_image_path = "exp73"

    #SÆT TIL TRUE HVIS DU VIL KØRE ALLE BILLEDER SÅ RUNS MAPPE BLIVER LAVET
    #SÆT TIL FALSE HVIS DU HAR RUNS MAPPERNE
    runAgain(doImages=False)

    # FOLDER_PATH = KUN KØR FOR EN I STEDET FOR ALLE, F.EKS. folder_path=test_image_path
    # Vil du køre alle billederne skal du bare ikke specificere folder_path
    data_with_digit_9 = getNineLabels(parent_directory, folder_path=test_image_path)
    
    process_image_data(data_with_digit_9)

if __name__ == "__main__":
    main()