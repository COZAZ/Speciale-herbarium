import os
from difflib import SequenceMatcher
from YOLO.label_detection import get_label_info, evaluate_label_detection_performance
from OCR.output_handler import evaluate_craft_ocr
from BERT.text_data_synthesizer import synthesize_text_data
from BERT.pred import parse_ocr_text

# TODO: add score for BERT
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
        print("\nRunning accuracy test for OCR output")
        ocr_score, text_amount = evaluate_craft_ocr()
        print("OCR text prediction accuracy: {0}%".format(round(ocr_score, 3)))
        print("Tested on {0} labels".format(text_amount))
    else:
        print("OCR output text does not exist. To compute OCR (CRAFT) accuracy, please run OCR on your images")
    
    # BERT accuracy
    testBERTAccuracy()
    
# BERT model accuracy
# TODO: Move BERT testing to BERT folder
def testBERTAccuracy():
    print("\nRunning accuracy test for BERT model")
    dataPoints = 100
    trueText_validation = synthesize_text_data(dataPoints, asJson=False)
    trueText_JSONized = []

    # Convert text to (stylized)JSON format
    counter = 1
    for obj in trueText_validation:
        obj_text = obj["tokens"]
        obj_json = {"image": str(counter) + ".jpg", "label_type": "X", "text": obj_text}

        trueText_JSONized.append(obj_json)
        counter += 1
    
    # Index values info:
    # 0 - Image name
    # 1 - Specimen
    # 2 - Location
    # 3 - Leg
    # 4 - Det
    # 5 - Date
    # 6 - Coords
                    
    predText_validation = parse_ocr_text(trueText_JSONized, True)
    label_score = [["B-SPECIMEN", 1, 0], ["B-LOCATION", 2, 0], ["B-LEG", 3, 0], ["B-DET", 4, 0], ["B-DATE", 5, 0], ["B-COORD", 6, 0]]
    
    for i in range(dataPoints):
        current_true_text = trueText_validation[i]
        current_pred_text = predText_validation[i]

        # TODO: Ask Kim how we should compare the different labels
        for elm in label_score:
            true_token = extract_token(current_true_text, elm[0])
            pred_token = current_pred_text[elm[1]]

            current_similarity = SequenceMatcher(None, true_token, pred_token).ratio()
            elm[2] += current_similarity

    print("Tested on {0} text objects".format(dataPoints))
    for elm in label_score:
        label_type = elm[0]
        label_score = round((elm[2] / dataPoints) * 100, 2)
        print("General {0} similarity score: {1}%".format(label_type, label_score))

    print("\nTesting complete")

def extract_token(text, label_type):
    # Find index of 'B-label_type' in labels, where label_type could be 'SPECIMEN', 'Date' etc.
    label_index = text["labels"].index(label_type)
    # Find the corresponding token
    label_token = text["tokens"][label_index]

    return label_token

runTests()