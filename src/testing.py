import os
import pandas as pd
from YOLO.label_detection import get_label_info, evaluate_label_detection_performance, compute_avr_conf
from OCR.output_handler import evaluate_craft_ocr
from BERT.testing_BERT import testBERTAccuracy
from final_output_test import compare_csv

def runTests():
    ### Show performance scores of all components ###
    print("Running performance tests...\n")

    image_directory = "linas_images"
    parent_directory = image_directory + "_runs"

    image_dic_conf = "herb_images_1980_runs"

    institute_label_data = None
    annotation_label_data = None

    institute_accuracy = 0
    annotation_accuracy = 0
    ocr_score = 0

    # YOLO box accuracy
    if os.path.exists(parent_directory):
        print("### Running accuracy test for YOLO labels ###")
        institute_label_data, annotation_label_data = get_label_info(parent_directory)
        institute_accuracy, annotation_accuracy = evaluate_label_detection_performance(institute_label_data, annotation_label_data)

        if institute_accuracy != None:
            print(" - Institution label prediction accuracy: {0}%".format(round(institute_accuracy, 3)))
            print(" (Tested on {0} institutional labels)".format(len(institute_label_data)))
        else: print("No institutional labels found.")

        if annotation_accuracy != None:
            print(" - Annotation label prediction accuracy: {0}%".format(round(annotation_accuracy, 3)))
            print(" (Tested on {0} annotation labels)".format(len(annotation_label_data)))
        else: print("No annotation labels found.") 
    else:
        print("Labeled Linas images do not exist. To compute YOLO accuracy, please generate labels for them with YOLO")
    
    # YOLO confidence score
    if os.path.exists(image_dic_conf):
        print("\n### Running average YOLO confidence score ###")
        i_conf, a_conf, total_conf  = compute_avr_conf(image_dic_conf)
        print(" - Average YOLO institutional label confidence score: {0}%".format(round(i_conf[0]*100, 2)))
        print(" (Tested on {0} institutional labels)".format(i_conf[1]))
        print(" - Average YOLO annotation label confidence score: {0}%".format(round(a_conf[0]*100, 2)))
        print(" (Tested on {0} annotation labels)".format(a_conf[1]))

        print(" - Combined average confidence score: {0}%".format(round(total_conf*100, 2)))
    else:
        print("Labeled machine images not found. Make sure to apply YOLO model on these images")

    # OCR accuracy
    if os.path.exists("../ocr_output_test.json"):
        print("\n### Running accuracy test for OCR output ###")
        ocr_score, text_amount = evaluate_craft_ocr()
        print(" - OCR text prediction accuracy: {0}%".format(round(ocr_score, 3)))
        print(" (Tested on {0} labels)".format(text_amount))
    else:
        print("OCR output text does not exist. To compute OCR (CRAFT) accuracy, please run OCR on your images")
    
    # BERT accuracy
    print("\n### Running accuracy test for BERT model ###")
    data_points = 1000
    label_scores, total_score, specimen_count, location_count, leg_count, det_count, date_count, coord_count, hardcases, easycases = testBERTAccuracy(data_points)
    for elm in label_scores:
        print("  - General {0} similarity score: {1}%".format(elm[0], elm[1]))
        
    print(" - Overall BERT model accuracy: {0}%".format(total_score))
    print(" (Tested on {0} text objects)".format(data_points))

    """
    print("\nCounting correct class matches...")
    print("Correct Specimen count low threshold: {0}/{1}".format(specimen_count[1], specimen_count[2]))
    print("Correct Specimen count high threshold: {0}/{1}".format(specimen_count[0], specimen_count[2]))

    print("\nCorrect Location count low threshold: {0}/{1}".format(location_count[1], location_count[2]))
    print("Correct Location count high threshold: {0}/{1}".format(location_count[0], location_count[2]))

    print("\nCorrect Leg count low threshold: {0}/{1}".format(leg_count[1], leg_count[2]))
    print("Correct Leg count high threshold: {0}/{1}".format(leg_count[0], leg_count[2]))

    print("\nCorrect Det count low threshold: {0}/{1}".format(det_count[1], det_count[2]))
    print("Correct Det count high threshold: {0}/{1}".format(det_count[0], det_count[2]))

    print("\nCorrect Date count low threshold: {0}/{1}".format(date_count[1], date_count[2]))
    print("Correct Date count high threshold: {0}/{1}".format(date_count[0], date_count[2]))

    print("\nCorrect Coordinate count low threshold: {0}/{1}".format(coord_count[1], coord_count[2]))
    print("Correct Coordinate count high threshold: {0}/{1}".format(coord_count[0], coord_count[2]))
    """

    # Comparison of final CSV files
    bert_csv = pd.read_csv("../herbarium_BERT.csv")
    post_csv = pd.read_csv("../herbarium_post.csv")
    true_csv = pd.read_csv("../herb_images_1980_true.csv", sep=',', quotechar='"', skipinitialspace=True)

    avg_specimen_similarity, avg_location_similarity, avg_legit_similarity, avg_determinant_similarity, avg_date_similarity, avg_coordinates_similarity, avg_total = compare_csv(post_csv, true_csv)

    print("\n### Testing final CSV database output ###")
    print("Using Fuzz string similarity:")
    print(" - Average Specimen ground truth CSV Similarity: {0}%".format(round(avg_specimen_similarity, 2)))
    print(" - Average Location grounf truth CSV Similarity: {0}%".format(round(avg_location_similarity, 2)))
    print(" - Average Legit ground truth CSV Similarity: {0}%".format(round(avg_legit_similarity, 2)))
    print(" - Average Determinant ground truth CSV Similarity: {0}%".format(round(avg_determinant_similarity, 2)))
    print(" - Average Date ground truth CSV Similarity: {0}%".format(round(avg_date_similarity, 2)))
    print(" - Average Coordinates ground truth CSV Similarity: {0}%".format(round(avg_coordinates_similarity, 2)))

    print("\n - Average ground truth total Fuzz similarity: {0}%".format(round(avg_total, 2)))

    print("\nUsing character error:")
    print("TBD")

runTests()