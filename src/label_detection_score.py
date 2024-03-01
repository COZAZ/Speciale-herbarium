import glob as g
import numpy as np

def Evaluate_label_detection_performance(label_data, detail=False):

    correct_boxes = 0

    # Tolerance level of 5%.
    # This means that a correctly predicted box is within 5% of the true box location
    tolerance = 0.05

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

    accuracy = (correct_boxes / len(label_data)) * 100

    return accuracy