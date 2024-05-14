import os
from YOLO.label_detection import get_label_info, predict_labels
from OCR.character_recognizer import process_image_data
from OCR.output_handler import save_ocr_output
from BERT.bert_to_csv import createCSV

def main():

    image_directory = "herb_images_1980"

    run_all = True

    if run_all and os.path.exists("../" + image_directory):
        print("\nRunning pipeline...")

        ### Perform label detection ###
        print("\nPredicting labels...")
        #predict_labels(image_directory)
        #institute_label_data, annotation_label_data = get_label_info(labels_directory)

        ### Perform OCR ###
        print("\nPerforming OCR on images...")
        #processed_images_data = process_image_data(institute_label_data, annotation_label_data, image_directory)
        #save_ocr_output(processed_images_data)

        ### Perform NER to parse output text ###
        print("\nParsing text to output...")
        createCSV()

        print("\nPipeline process complete")
    
    else:
        print("Error: invalid image collection path")

if __name__ == "__main__":
    main()