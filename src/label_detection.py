import glob as g
import numpy as np
import os
from yolov5.detect import run
from pathlib import Path

# get the path/directory
#folder_dir = "../herb_images"
folder_dir = "../linas_images"
 
# Iterate over files in that directory
# Get a list of all JPEG files found
images = list(Path(folder_dir).glob('*.jpg'))

# Now, 'images' contains the filenames sorted numerically
def predict_labels(doImages=False):
    if doImages:
            run(
            weights="MELU-Trained-ObjDetection-Model-Yolov5-BEST.pt",
            source=folder_dir,
            conf_thres=0.4,
            imgsz=(416, 416),
            nosave = False,
            view_img = False,
            save_txt = True,
            project = 'runs'
            )

def get_label_info(parent_directory, image_file_extension=".jpg", test_images=None):
    all_data_with_digit_9 = []
    all_data_with_digit_3 = []

    exp_folder = Path(parent_directory) / "exp"
    labels_folder = exp_folder / "labels"

    # Check if the labels folder exists
    if labels_folder.exists() and labels_folder.is_dir():

        if test_images == None:
            label_files = list(labels_folder.glob('*.txt'))
        else:
            specific_files = []
            for name in test_images:
                path = Path('runs/exp/labels/' + name)
                specific_files.append(path)
            
            label_files = specific_files
            
        for label_file in label_files:
            with open(label_file, 'r') as file:
                lines = file.readlines()

            for line in lines:
                columns = line.strip().split()
                if columns:
                    image_file_name = label_file.stem + image_file_extension
                    if columns[0] == '9':
                        all_data_with_digit_9.append((columns, image_file_name))
                    elif columns[0] == '3':
                        all_data_with_digit_3.append((columns, image_file_name))

    return all_data_with_digit_9, all_data_with_digit_3
   
def evaluate_label_detection_performance(institute_data, annotation_data, detail=True):
    institute_label_data = institute_data
    annotation_label_data = annotation_data

    # Type "i": institution boxes will be compared
    # Type "a": annotation boxes will be compared
    institute_accuracy = compare_bounding_boxes(institute_label_data, label_type="i")
    annotation_accuracy = compare_bounding_boxes(annotation_label_data, label_type="a")

    return institute_accuracy, annotation_accuracy

def compare_bounding_boxes(label_data, label_type=None):

    correct_boxes = 0

    for predicted_box in label_data:

        # ID name of image
        im_name = predicted_box[1][:-4]

        # Predicted bounding box coordinates from YOLO model
        predicted_coordiantes = np.array(predicted_box[0][1:]).astype(float)

        # Create a pattern to match all .txt files in the folder
        file_pattern = "../true_annotations/" + im_name + ".txt"
        file_paths = g.glob(file_pattern)

        if not file_paths:
            print("No files found for ID: {0}".format(im_name))
        else:
            f_path = file_paths[0]

            # Read the matching .txt file with the true box location
            with open(f_path, 'r') as file:
                content = file.readlines()

                if label_type is not None:
                    if label_type == "i":
                        # Filter lines starting with 0 (institutional labels)
                        content = [line for line in content if line.startswith('0')]
                    elif label_type == "a":
                        # Filter lines starting with 1 (annotation labels)
                        content = [line for line in content if line.startswith('1')]

                true_boxes = np.array([np.array(line.strip().split()[1:]).astype(float) for line in content])

                pred_box_coords = extract_corner_points(predicted_coordiantes)

                for true_box in true_boxes:
                    true_box_coords = extract_corner_points(true_box)
                    #print("True box coords:", true_box_coords)
                    #print("Predicted box coords:", pred_box_coords)

                    iou = get_intersection_over_union(pred_box_coords, true_box_coords)
                    #print("pred_box_coords:", pred_box_coords)
                    #print("true_box_coords:", true_box_coords)                       
                    #print("IOU:", iou)

                    if iou >= 0.50:
                        correct_boxes += 1

    return (correct_boxes / len(label_data)) * 100

def get_intersection_over_union(boxA, boxB):
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0][0], boxB[0][0])
    yA = max(boxA[0][1], boxB[0][1])
    xB = min(boxA[3][0], boxB[3][0])
    yB = min(boxA[3][1], boxB[3][1])
    
    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    # Compute the area of both the boxes
    boxAArea = abs(boxA[1][0] - boxA[0][0]) * abs(boxA[2][1] - boxA[0][1])
    boxBArea = abs(boxB[1][0] - boxB[0][0]) * abs(boxB[2][1] - boxB[0][1])
    
    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou

def extract_corner_points(coords_set):
    center_x = coords_set[0]
    center_y = coords_set[1]
    width = coords_set[2]
    height = coords_set[3]

    # Calculate half-width and half-height
    half_width = width / 2
    half_height = height / 2

    # Calculate corner points
    top_left = (center_x - half_width, center_y - half_height)
    top_right = (center_x + half_width, center_y - half_height)
    bottom_left = (center_x - half_width, center_y + half_height)
    bottom_right = (center_x + half_width, center_y + half_height)

    box_coords = [top_left, top_right, bottom_left, bottom_right]

    return box_coords