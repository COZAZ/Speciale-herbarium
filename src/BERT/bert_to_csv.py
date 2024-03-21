# Hello Jonathan. This file is for transforming BERT output to CSV. I SEE COMMENT. DO YOU SEEM MY REPLY????
from BERT.pred import parse_ocr_text
import csv

# Create a CSV file that contains the parsed OCR text
def createCSV():
    print("Creating CSV...")

    interests = parse_ocr_text() # TODO: at some point, load ocr output JSON as text data

    data = interests.copy()
    data.insert(0,['Specimen', 'Location', 'Legit', 'Determinant', 'Date', 'Coordinates'])

    with open("herbarium.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in data:
            csv_writer.writerow(row)

    print("CSV created")