import glob as g
import numpy as np

def compute_bounding_box_score(label_data):

    correct_boxes = 0

    # Tolerance level of 5%.
    # This means that a correctly predicted box is within 5% of the true box location
    tolerance = 0.05

    for label_object in label_data:

        # ID name of image
        im_name = (label_object[1])[:-4]

        # Predicted bounding box coordinates from YOLO model
        predicted_coordiantes = np.array((label_object[0])[1:]).astype(float)

        # Create a pattern to match all .txt files in the folder
        file_pattern = "../linas_true_annotations/" + im_name + ".txt"
        file_paths = g.glob(file_pattern)

        if not file_paths:
            print("No files found for the current ID.")
        else:
            f_path = file_paths[0]

            # Read the matching .txt file with the true box location
            with open(f_path, 'r') as file:
                content = ((file.read())[1:]).split()
                true_coordinates = (np.array(content)).astype(float)

                #print("({0}) True box cooridnates:".format(im_name))
                #print(true_coordinates)

                #print("({0}) YOLO prediction coordinates:".format(im_name))
                #print(predicted_coordiantes)

                ### Computation of coordinate differences ###
                true_center_point = np.array([true_coordinates[0], true_coordinates[1]])
                predicted_center_point = np.array([predicted_coordiantes[0], predicted_coordiantes[1]])

                true_width = true_coordinates[2]
                predicted_width = predicted_coordiantes[2]

                true_height = true_coordinates[3]
                predicted_height = predicted_coordiantes[3]

                center_dist = np.linalg.norm(true_center_point - predicted_center_point)
                ###

                #print("center diff:", center_dist)

                # Accept YOLO predicted box if within 5% of true box
                if ((center_dist < tolerance)
                    and (np.abs(true_width - predicted_width) < tolerance)
                    and (np.abs(true_height - predicted_height) < tolerance)):

                    correct_boxes += 1

    # Final YOLO performance score
    score = (correct_boxes / len(label_data)) * 100

    return score