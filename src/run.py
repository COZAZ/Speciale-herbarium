import argparse
import os
from YOLO.label_detection import get_label_info, predict_labels
from OCR.character_recognizer import process_image_data
from OCR.output_handler import save_ocr_output
from BERT.text_data_synthesizer import synthesize_text_data
from BERT.preBERT import train_bert
from BERT.bert_to_csv import createCSV

# TODO: Clean up the code and remove unnecessary files
def main(yolo=False, ocr=False, bert=False):
    print("Starting pipeline...")
    machine = ["689351.txt", "704605.txt", "859622.txt"]
    #linas = ["682156.txt", "682897.txt"]

    test_specific_paths = machine
    image_directory = "machine_images_color"
    parent_directory = image_directory + "_runs"

    institute_label_data = None
    annotation_label_data = None
    ocr_is_ready = False

    run_all = True

    if yolo and (not os.path.exists(parent_directory)):
        ### PIPELINE step 1: Identify bounding boxes ###
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
            institute_label_data, annotation_label_data = get_label_info(parent_directory)
            ocr_is_ready = True

            print("Image labels exist ({0} institutional labels and {1} annotation labels), skipping label detection".format(len(institute_label_data), len(annotation_label_data)))
    
    if ocr and ocr_is_ready:
        ### PIPELINE step 3: Extract text from images ###
        # Performs OCR on cropped images according to the predicted bounding box locations
        processed_images_data = process_image_data(institute_label_data, annotation_label_data, image_directory)
        save_ocr_output(processed_images_data)
        run_all = False
    else:
        if (not ocr_is_ready): print("Error: System not ready for OCR yet. Make sure you have the YOLO-labeled images before you set --ocr")
        elif not os.path.exists("ocr_output.json"):
            print("Warning: No saved OCR output found, please perform OCR by using the flag --ocr when calling run.py")
            print("Pipeline will continue without OCR results...")
        else:
            print("OCR output exists, skipping OCR")

    if bert:
        ### PIPELINE step 4: Parse text results from OCR ###
        # Generate text for training BERT model
        #TODO: We need to train the model once and for all. Then we can remove the --bert flag
        print("Generating training text for BERT model...")
        number_of_text = 50
        synthesize_text_data(number_of_text, asJson=True)
        print("Text generation done")

        # Training the BERT model
        print("Training BERT model with {0} text objects...".format(number_of_text))
        train_bert()
        print("Training complete")

        run_all = False
    else:
        if not os.path.exists("BERT_MODEL"):
            print("Warning: BERT model not trained yet, please use flag --bert when calling run.py")
            print("Pipeline process will continue without BERT model...")
            #run_all = False
        else:
            print("BERT model exists, skipping training")

    if run_all:
        print("\nRunning Analysis...")

        # Use BERT model to parse image text into a resulting CSV file
        createCSV()

        print("Pipeline process complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script is for running the pipeline system. Necessary data must be generated first, by using the below flags.")
    parser.add_argument("--yolo", action="store_true", help="Generate labels on images with YOLO model")
    parser.add_argument("--ocr", action="store_true", help="Perform OCR on the YOLO-labeled images")
    parser.add_argument("--bert", action="store_true", help="Generate OCR-like text strings and train the BERT model (only text gen currently)")

    args = parser.parse_args()

    main(args.yolo, args.ocr, args.bert)