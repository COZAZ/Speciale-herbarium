import json
import csv
import unicodedata
import re
from BERT.pred import parse_ocr_text
from thefuzz import fuzz

# Create a CSV file that contains the parsed OCR text
def createCSV():
    print("Creating CSV...")

    predicted = parse_ocr_text()

    data = predicted.copy()
    data.insert(0, ['Catalog number', 'Specimen', 'Location', 'Legit', 'Determinant', 'Date', 'Coordinates'])

    with open("ocr_output.json", 'r') as f:
        ocr_text = json.load(f)

    for i, elm in enumerate(predicted):
        if ocr_text[i]["text"] == 1:
            continue
    
        threshold = 0.75
        ocr_object = ocr_text[i]["text"]

        pred_spec = elm[1]
        spec_score = 0
        spec_correct_index = 0
        
        pred_loc = elm[2]
        loc_score = 0
        loc_correct_index = 0
        
        pred_leg = elm[3]
        leg_score = 0
        leg_correct_index = 0
        
        pred_det = elm[4]
        det_score = 0
        det_correct_index = 0
        
        pred_date = elm[5]
        date_score = 0
        date_correct_index = 0
        
        pred_coord = elm[6]
        coord_score = 0
        coord_correct_index = 0

        for j, text_piece in enumerate(ocr_object):
            text_piece = text_piece.replace(" ", "")
            pred_spec = pred_spec.replace(" ", "")
            pred_loc = pred_loc.replace(" ", "")
            pred_leg = pred_leg.replace(" ", "")
            pred_det = pred_det.replace(" ", "")
            pred_date = pred_date.replace(" ", "")
            pred_coord = pred_coord.replace(" ", "")

            # Find matching specimen from OCR
            spec_calc = fuzz.ratio(text_piece, pred_spec)
            if spec_calc > spec_score:
                spec_score = spec_calc
                spec_correct_index = j
            
            loc_calc = fuzz.ratio(text_piece, pred_loc)
            if loc_calc > loc_score:
                loc_score = loc_calc
                loc_correct_index = j

            leg_calc = fuzz.ratio(text_piece, pred_leg)
            if leg_calc > leg_score:
                leg_score = leg_calc
                leg_correct_index = j

            det_calc = fuzz.ratio(text_piece, pred_det)
            if det_calc > det_score:
                det_score = det_calc
                det_correct_index = j

            date_calc = fuzz.ratio(text_piece, pred_date)
            if date_calc > date_score:
                date_score = date_calc
                date_correct_index = j

            coord_calc = fuzz.ratio(text_piece, pred_coord)
            if coord_calc > coord_score:
                coord_score = coord_calc
                coord_correct_index = j
        
        #print("Predicted specimen:", pred_spec)
        #print("Found OCR specimen:", ocr_object[spec_correct_index])

        if (spec_score >= threshold):
            spec_csv = ocr_object[spec_correct_index]

            matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[a-z]+)*\b', spec_csv)
            spec_filter = [match for match in matches if is_species_name(match)]

            if (len(spec_filter) != 0) and (len(spec_filter[0]) <= 50):
                #print("Current pred:", pred_spec)
                #print("Current test: ", spec_filter)
                temp_spec = spec_filter[0].split()
                if len(temp_spec) > 3:
                    data[i+1][1] = pred_spec

                else:
                    print(spec_filter)
                    data[i+1][1] = spec_filter[0]

            else:
                data[i+1][1] = pred_spec
            
        if loc_score >= threshold:
            loc_csv = ocr_object[loc_correct_index]
            data[i+1][2] = loc_csv

        if leg_score >= threshold:
            leg_csv = ocr_object[leg_correct_index]
            data[i+1][3] = leg_csv

        if det_score >= threshold:
            det_csv = ocr_object[det_correct_index]
            data[i+1][4] = det_csv

        if date_score >= threshold:
            date_csv = ocr_object[date_correct_index]
            data[i+1][5] = date_csv

        if coord_score >= threshold:
            coord_csv = ocr_object[coord_correct_index]
            data[i+1][6] = coord_csv        

    with open("herbarium_lookback.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in data:
            clean_row = []
            for string in row:
                current_clean_string = unicodedata.normalize("NFKD", string).encode("ascii", "replace").decode()
                clean_row.append(current_clean_string)
            csv_writer.writerow(clean_row)

    print("CSV created")

def is_species_name(s):
    words = s.split()
    return len(words) >= 2 and words[0].istitle() and all(word.islower() or word.istitle() for word in words[1:])
