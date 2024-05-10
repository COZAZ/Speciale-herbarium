from YOLO.label_detection import get_label_info, predict_labels
from OCR.character_recognizer import process_image_data
from OCR.output_handler import save_ocr_output
from BERT.bert_to_csv import createCSV

def main():
    print("Starting pipeline...")
    machine = ["689351.txt", "704605.txt", "859622.txt"]

    test_specific_paths = machine
    image_directory = "herb_images_1980"
    labels_directory = image_directory + "_runs"

    run_all = True

    if run_all:
        print("\nRunning Analysis...")

        ### Perform label detection ###
        predict_labels(image_directory)
        institute_label_data, annotation_label_data = get_label_info(labels_directory)

        ### Perform OCR ###
        processed_images_data = process_image_data(institute_label_data, annotation_label_data, image_directory)
        save_ocr_output(processed_images_data)

        ### Perform NER to parse output text ###
        createCSV()

        print("\nPipeline process complete")

if __name__ == "__main__":
    main()