import cv2
import easyocr

def resize_image(image, scale=35):
    scale_percent = scale  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    
    return resized_image

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
    result = reader.readtext(cropped_image, paragraph=True)

    # UNCOMMENT IF YOU WANT TO SEE THE CROPPED RESULT
    #resized_cropped_image = resize_image(cropped_image, scale=35)
    #cv2.imshow('Cropped Image', resized_cropped_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return result

def process_image_data(institute_label_data, annotation_label_data, im_dir):
    """Process image data."""

    progress_counter = 0
    total_labels = len(institute_label_data) + len(annotation_label_data)

    institute_predicted_text, current_counter = perform_ocr(institute_label_data, "i", im_dir, progress_counter, total_labels)
    annotation_predicted_text, _ = perform_ocr(annotation_label_data, "a", im_dir, current_counter, total_labels)

    return institute_predicted_text + annotation_predicted_text

def perform_ocr(label_data, annotation_type, im_dir, counter, label_total):
    ocr_results = []

    for label in label_data:
        image_path = "../" + im_dir + '/' + label[1]
        image, image_size = load_and_preprocess_image(image_path)

        if image is None:
            continue

        coordinates = label[0][1:5]
        bbox = calculate_coordinates(image_size, coordinates)

        # UNCOMMENT IF YOU WANT TO SEE THE DRAWN BOUNDING BOX
        #display_image_with_bbox(image, bbox)

        predicted_text = process_cropped_image(image, bbox)

        processed_image_info_annotate = (label[1], annotation_type, predicted_text)

        ocr_results.append(processed_image_info_annotate)

        counter += 1
        
        print("Labels processed with OCR: {0}/{1}".format(counter, label_total))
    
    return ocr_results, counter