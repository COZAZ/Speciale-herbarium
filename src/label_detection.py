import glob as g
import numpy as np
from yolov5.detect import run
from pathlib import Path

# get the path/directory
#folder_dir = "../herb_images"
folder_dir = "../linas_images"
 
# Iterate over files in that directory
# Get a list of all JPEG files found
images = list(Path(folder_dir).glob('*.jpg'))

# Now, 'images' contains the filenames sorted numerically
def predictLabels(doImages=False):
    if doImages:
        for image in images:
            run(
            weights="MELU-Trained-ObjDetection-Model-Yolov5-BEST.pt",
            source=image,
            #source="../herb_images/728989.jpg",
            #source="../herb_images/697008.jpg",
            #source="../linas_images/732177.jpg",
            #source="../linas_images/682156.jpg",
            conf_thres=0.4,
            imgsz=(416, 416),
            nosave = False,
            view_img = False,
            save_txt = True,
            project = 'runs'
            )

def getInstituteLabels(parent_directory, folder_pattern="exp*", folder_paths=None, image_file_name=None):
    # Initialize the list to store data from all label files
    all_data_with_digit_9 = []

    # Determine the folders to iterate over
    if folder_paths:
        folders = [Path(parent_directory) / folder_path for folder_path in folder_paths]
    else:
        folders = Path(parent_directory).glob(folder_pattern)

    # Iterate through each folder
    for folder in folders:
        print(f"Processing folder: {folder}")
        # Access the 'labels' folder within the folder
        labels_folder = folder / "labels"
        
        # Check if the 'labels' folder exists
        if labels_folder.exists() and labels_folder.is_dir():
            print(f"Labels folder found: {labels_folder}")
            # Find the label file within the 'labels' folder
            label_files = list(labels_folder.glob('*.txt'))
            print(f"Label files: {label_files}")

            # Process each label file
            for label_file in label_files:
                # Read the contents of the label file
                with open(label_file, 'r') as file:
                    lines = file.readlines()

                # Extract data where the first column is a digit 9
                data_with_digit_9 = []
                for line in lines:
                    # Split the line by spaces assuming it's in a format like '9 x y width height'
                    columns = line.strip().split()
                    if len(columns) > 0 and columns[0] == '9':
                        # If image_file_name is provided, use it
                        if image_file_name:
                            image_file = labels_folder.parent / image_file_name
                        else:
                            # Otherwise, try to find any image file in the parent folder
                            image_files = list(labels_folder.parent.glob('*.jpg'))
                            if image_files:
                                image_file = image_files[0]
                            else:
                                print(f"No image file found in {labels_folder.parent}")
                                continue

                        # Append a tuple containing the coordinates and the image file name
                        data_with_digit_9.append((columns, image_file.name))

                # Append data from this label file to the list
                all_data_with_digit_9.extend(data_with_digit_9)

    return all_data_with_digit_9

def Evaluate_label_detection_performance(label_data, detail=False):

    correct_boxes = 0

    # Tolerance level of 10%.
    # This means that a correctly predicted box is within 10% of the true box location
    tolerance = 0.1

    for predicted_box in label_data:

        # ID name of image
        im_name = (predicted_box[1])[:-4]

        # Predicted bounding box coordinates from YOLO model
        predicted_coordiantes = np.array((predicted_box[0])[1:]).astype(float)

        # Create a pattern to match all .txt files in the folder
        file_pattern = "../linas_annotations/" + im_name + ".txt"
        file_paths = g.glob(file_pattern)

        if not file_paths:
            print("No files found for the current ID.")
        else:
            f_path = file_paths[0]

            # Read the matching .txt file with the true box location
            with open(f_path, 'r') as file:
                content = np.array(file.readlines())
                content = np.array(list(map(lambda x: np.array(((x[2:-1]).split())).astype(float), content)))

                ### Coordinate-related computations to compare predicted box with true boxes ###
                true_coordinates = content[:, :4]
                true_center_point = true_coordinates[:, :2]
                predicted_center_point = predicted_coordiantes[:2]

                true_width = true_coordinates[:, 2]
                predicted_width = predicted_coordiantes[2]

                true_height = true_coordinates[:, 3]
                predicted_height = predicted_coordiantes[3]

                center_dist = np.linalg.norm(true_center_point - predicted_center_point, axis=1)

                center_checks = center_dist <= tolerance
                width_checks = np.abs(true_width - predicted_width) < tolerance
                height_checks = np.abs(true_height - predicted_height) < tolerance

                true_box_checks = np.all([center_checks, width_checks, height_checks], axis=0)

                ###

                if detail:
                    print("Processing predicted box for image:", im_name + ".jpg")
                    print("Center match:", center_checks)
                    print("Width match:", width_checks)
                    print("Height match:", height_checks)
                    print("True box match:", true_box_checks)
                    print("\n")
                
                if np.any(true_box_checks):
                    correct_boxes += 1
                else:
                    print("Failed to classify predicted box for image:", im_name + ".jpg")

    accuracy = (correct_boxes / len(label_data)) * 100

    return accuracy