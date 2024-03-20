import pygbif.species as gb
import argparse
import os
from OCR.character_recognizer import process_image_data
from OCR.output_handler import save_ocr_output
from YOLO.label_detection import get_label_info, predict_labels, evaluate_label_detection_performance
from BERT.text_data_synthesizer import synthesize_text_data, pretty_print_text_data

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

def main(yolo=False, ocr=False, bert=False):
    parent_directory = "runs"
    test_specific_paths = ["689351.txt", "704605.txt"]
    image_directory = "herb_images_machine"

    institute_label_data = None
    annotation_label_data = None
    ocr_is_ready = False

    run_all = True

    if yolo and (not os.path.exists(parent_directory)):
        ### PIPELINE step 1: Identify bounding boxes ###
        # Set doImages to True to predict labels of all images in herb_images
        # Set to false to skip this step, if you already have the "runs" results
        predict_labels(image_directory)
        run_all = False
    else:
        if not os.path.exists(parent_directory):
            print("Error: YOLO labels not generated yet, please use flag --yolo when calling run.py")
            run_all = False
        else:
            ### PIPELINE step 2: Find label locations in images ###
            # Set folder_path to specific exp folder to get label location of only one image
            # To run on all images, do not set the folder_path parameter
            institute_label_data, annotation_label_data = get_label_info(parent_directory, test_images=test_specific_paths)
            #institute_accuracy, annotation_accuracy = evaluate_label_detection_performance(institute_label_data, annotation_label_data)

            ocr_is_ready = True

            #print("\nInstitution label prediction accuracy: {0}%".format(institute_accuracy))
            #print("Annotation label prediction accuracy: {0}%".format(annotation_accuracy))

            print("Image labels exist ({0} instituional labels and {1} annotation labels), skipping label detection".format(len(institute_label_data), len(annotation_label_data)))
    
    if ocr and (not os.path.exists("../ocr_output.json")) and ocr_is_ready:
        ### PIPELINE step 3: Extract text from images ###
        # Performs OCR on cropped images according to the predicted bounding box locations
        processed_images_data = process_image_data(institute_label_data, annotation_label_data, image_directory)
        save_ocr_output(processed_images_data)
        run_all = False
    else:
        if not ocr_is_ready: print("Error: System not ready for OCR yet. Make sure you have the YOLO-labeled images before you set --ocr")
        elif not os.path.exists("../ocr_output.json"):
            print("Error: No saved OCR output found, please perform OCR by using the flag --ocr when calling run.py")
            run_all = False
        else:
            print("OCR output exists, skipping OCR")

    if bert:
        ### PIPELINE step 4: Parse text results from OCR ###
        # Generate text for training BERT model
        print("Generating training text for BERT model...")
        generated_bert_text = synthesize_text_data(30, asJson=True)
        #pretty_print_text_data(generated_bert_text)

        ### TODO: Train the BERT model here??? ###






        print("Text generation done")

        run_all = False
    else:
        if not os.path.exists("synth_data.json"):
            print("Error: BERT model not trained yet, please use flag --bert when calling run.py")
            run_all = False
        else:
            print("Artificial text exists, skipping generation")

    if run_all:
        print("Running Analysis...")

        #ocr_accuracy = 1

        #print(processed_images_data)
        #print("OCR (CRAFT) performance accuracy: {0}%".format(ocr_accuracy))

        ### GBIF STUFF BELOW ###

        #found_plant_names = findNames(processed_images_data)
        #match_rate = compute_name_precision(found_plant_names)

        #print(found_plant_names)
        #print("Match rate from running {0} images: {1}%".format(len(found_plant_names), match_rate*100))

        print("Pipeline process complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script is for running the pipeline system. Necessary data must be generated first, by using the below flags.")
    parser.add_argument("--yolo", action="store_true", help="Generate labels on images with YOLO model")
    parser.add_argument("--ocr", action="store_true", help="Perform OCR on the YOLO-labeled images")
    parser.add_argument("--bert", action="store_true", help="Generate OCR-like text strings and train the BERT model (only text gen currently)")

    args = parser.parse_args()

    main(args.yolo, args.ocr, args.bert)