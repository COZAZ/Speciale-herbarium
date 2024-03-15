import pygbif.species as gb
import argparse
import os
from YOLO.label_detection import get_label_info, predict_labels, evaluate_label_detection_performance
from OCR.character_recognizer import process_image_data
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

def main(makeLabels=False, makeText=False):
    parent_directory = "runs"
    test_specific_paths = ["711477.txt", "711878.txt"]
    image_directory = "herb_images"

    run_all = True

    if makeLabels and (not os.path.exists(parent_directory)):
        ### PIPELINE step 1: Identify bounding boxes ###
        # Set doImages to True to predict labels of all images in herb_images
        # Set to false to skip this step, if you already have the "runs" results
        predict_labels(image_directory)
        run_all = False
    else:
        if not os.path.exists(parent_directory):
            print("Error: YOLO labels not generated yet, please use flag --makeLabels when calling run.py")
            run_all = False
        else:
            print("Runs folder exists, skipping label detection")
    
    if makeText:
        ### PIPELINE step 4: Parse text results from OCR ###
        # Generate text for training BERT model
        print("Generating training text for BERT model...")
        generated_bert_text = synthesize_text_data(30, asJson=True)
        pretty_print_text_data(generated_bert_text)
        print("Text generation done")

        run_all = False
    else:
        if not os.path.exists("synth_data.json"):
            print("Error: BERT training text not generated yet, please use flag --makeText when calling run.py")
            run_all = False
        else:
            print("Artificial text exists, skipping generation")

    if run_all:
        print("Running Analysis...")
        ### PIPELINE step 2: Find label location in images ###
        # Set folder_path to specific exp folder to get label location of only one image
        # To run on all images, do not set the folder_path parameter
        institute_label_data, annotation_label_data = get_label_info(parent_directory, test_images=test_specific_paths)
        #institute_accuracy, annotation_accuracy = evaluate_label_detection_performance(institute_label_data, annotation_label_data)

        #print("\nInstitution label prediction accuracy: {0}%".format(institute_accuracy))
        #print("Annotation label prediction accuracy: {0}%".format(annotation_accuracy))

        ### PIPELINE step 3: Extract text from images ###
        # Performs OCR on cropped images according to the predicted bounding box locations
        #processed_images_data = process_image_data(institute_label_data, annotation_label_data, image_directory)
        #print(processed_images_data)

        ### GBIF STUFF BELOW ###

        #found_plant_names = findNames(processed_images_data)
        #match_rate = compute_name_precision(found_plant_names)

        #print(found_plant_names)
        #print("Match rate from running {0} images: {1}%".format(len(found_plant_names), match_rate*100))

        print("Pipeline process complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script is for running the pipeline system. Necessary data must be generated first, by using the below flags.")
    parser.add_argument("--makeLabels", action="store_true", help="Generate labels on images with YOLO model")
    parser.add_argument("--makeText", action="store_true", help="Generate OCR-like text strings for training the BERT model")

    args = parser.parse_args()

    main(args.makeLabels, args.makeText)