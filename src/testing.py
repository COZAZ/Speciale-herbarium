import os
from YOLO.label_detection import get_label_info, evaluate_label_detection_performance
from OCR.output_handler import evaluate_craft_ocr
from BERT.testing_BERT import testBERTAccuracy

# TODO: Validation score done for BERT, but testing score missing
def runTests():
    ### Show performance scores of all components ###
    print("Running performance tests...\n")

    image_directory = "linas_images"
    parent_directory = image_directory + "_runs"

    institute_label_data = None
    annotation_label_data = None

    institute_accuracy = 0
    annotation_accuracy = 0
    ocr_score = 0

    # YOLO accuracy
    if os.path.exists(parent_directory):
        print("Running accuracy test for YOLO labels...")
        institute_label_data, annotation_label_data = get_label_info(parent_directory)
        institute_accuracy, annotation_accuracy = evaluate_label_detection_performance(institute_label_data, annotation_label_data)

        if institute_accuracy != None:
            print("Institution label prediction accuracy: {0}%".format(round(institute_accuracy, 3)))
            print("Tested on {0} institutional labels".format(len(institute_label_data)))
        else: print("No institutional labels found.")

        if annotation_accuracy != None:
            print("Annotation label prediction accuracy: {0}%".format(round(annotation_accuracy, 3)))
            print("Tested on {0} annotation labels".format(len(annotation_label_data)))
        else: print("No annotation labels found.") 
    else:
        print("Labeled Linas images do not exist. To compute YOLO accuracy, please generate labels for them with YOLO")

    # OCR accuracy
    if os.path.exists("../ocr_output.json"):
        print("\nRunning accuracy test for OCR output...")
        ocr_score, text_amount = evaluate_craft_ocr()
        print("OCR text prediction accuracy: {0}%".format(round(ocr_score, 3)))
        print("Tested on {0} labels".format(text_amount))
    else:
        print("OCR output text does not exist. To compute OCR (CRAFT) accuracy, please run OCR on your images")
    
    # BERT accuracy
    print("\nRunning accuracy test for BERT model...")
    data_points = 10
    bert_score = testBERTAccuracy(data_points)
    for elm in bert_score:
        label_type = elm[0]
        print("General {0} similarity score: {1}%".format(label_type, elm[2]))
    print("Tested on {0} text objects".format(data_points))

    print("\nTesting complete")
    
runTests()