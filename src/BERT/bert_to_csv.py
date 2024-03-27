from BERT.pred import parse_ocr_text
import csv

# Create a CSV file that contains the parsed OCR text
# TODO: Try to clean up wrong labels/text in CSV.
def createCSV():
    print("Creating CSV...")

    interests = parse_ocr_text()

    data = interests.copy()
    data.insert(0, ['Catalog number', 'Specimen', 'Location', 'Legit', 'Determinant', 'Date', 'Coordinates'])

    with open("herbarium.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in data:
            csv_writer.writerow(row)

    print("CSV created")