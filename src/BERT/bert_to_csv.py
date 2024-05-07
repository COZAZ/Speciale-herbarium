import json
import csv
import unicodedata
import re
import os
from BERT.pred import parse_ocr_text
from thefuzz import fuzz
from OCR.ocr_converter import AdjustTextLayout

# Create a CSV file that contains the parsed OCR text
def createCSV():
    print("Creating CSV...")

    if not (os.path.exists('ocr_sorted.json') and os.path.exists('ocr_post.json')):
        AdjustTextLayout()

    predicted = parse_ocr_text()

    data = predicted.copy()
    data.insert(0, ['Catalog number', 'Specimen', 'Location', 'Legit', 'Determinant', 'Date', 'Coordinates', "Label Type"])

    with open('ocr_post.json', 'r') as f:
        ocr_text = json.load(f)

    for i, elm in enumerate(predicted):
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
            # If Det never found in OCR reading
            if len(pred_det) == 0:
                det_score = 100

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
                    #print(spec_filter)
                    data[i+1][1] = spec_filter[0]

            else:
                data[i+1][1] = pred_spec
            
        if loc_score >= threshold:
            loc_csv = ocr_object[loc_correct_index]
            if "LOC" not in loc_csv.upper():
                # Search for string in ocr_text for "loc" and replace loc_csv with that string
                for k, text in enumerate(ocr_object):
                    if "LOC" in text.upper() and len(text) > 6:
                        loc_csv = text
                        break
                    if "LOC" in text.upper() and len(text) <= 5:
                        # log_csv should be text and the string after it
                        loc_csv = text + " " + ocr_object[k+1]
                        break

            if "LOC" in loc_csv.upper():
                # Find the index of "Loc"
                index = loc_csv.upper().find("LOC")
                # Extract the substring starting from the index of "Loc"
                loc_csv = loc_csv[index:]

            #print("pRE-loc:", loc_csv)
            #print("After loc: ", loc_csv)
            data[i+1][2] = loc_csv
                      
        if leg_score >= threshold:
            leg_csv = ocr_object[leg_correct_index]
            leg_csv = cleanName("LEG", ocr_object, leg_csv)
            leg_csv = findAndFormatLeg(leg_csv)
            
            det_index = leg_csv.upper().find("DET")
            if det_index != -1:
                leg_csv = leg_csv[:det_index]
                
            leg_csv = remove_digits(leg_csv)
            leg_csv = remove_date_month(leg_csv)
            leg_csv = ' '.join(leg_csv.split())

            if "LEG" not in leg_csv.upper():
                leg_csv = "Leg: " + leg_csv

            data[i+1][3] = leg_csv

        if det_score >= threshold:
            det_csv = ocr_object[det_correct_index]
            det_csv = cleanName("DET", ocr_object, det_csv)
            det_csv = ' '.join(det_csv.split())

            if "DET" not in det_csv.upper() or (pred_det == "") or ("DET" in det_csv.upper() and len(det_csv) <= 7):
               data[i+1][4] = findAndFormatDet(data[i+1][3])
            else:
                data[i+1][4] = findAndFormatDet(det_csv)
            
            if len(data[i+1][3]) <= 7:
                data[i+1][3] = data[i+1][4].replace("Det", "Leg")
                
        if date_score >= threshold:
            date_csv = ocr_object[date_correct_index]
            data[i+1][5] = date_csv

        if coord_score >= threshold:
            coord_csv = ocr_object[coord_correct_index]
            data[i+1][6] = coord_csv        

    data_slice = data[1:]
    sorted_data = sorted(data_slice, key=lambda x: int(x[0]))

    # We keep only one data entry per catalog number
    # since images with multiple institutional/annotation labels all describe the same plant
    unique_data = []
    encountered_catalog_numbers = []
    for entry in sorted_data:
        catalog_number = entry[0]
        if catalog_number not in encountered_catalog_numbers:
            unique_data.append(entry)
            encountered_catalog_numbers.append(catalog_number)

    with open("herbarium_lookback.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(data[0])
        for row in unique_data:
            clean_row = []
            #row = [row[0], row[3], row[4]]
            for string in row:
                current_clean_string = unicodedata.normalize("NFKD", string).encode("ascii", "replace").decode()
                clean_row.append(current_clean_string)
            csv_writer.writerow(clean_row)

    print("CSV created")

def is_species_name(s):
    words = s.split()
    return len(words) >= 2 and words[0].istitle() and all(word.islower() or word.istitle() for word in words[1:])

def cleanName(name_type, ocr_obj, name_text):
    if name_type not in name_text.upper():
        for text in ocr_obj:
            if name_type in text.upper() and len(text.replace(" ", "")) > 6:
                name_text = text
                break            
        
    if name_type in name_text.upper():
            # Find the index of "Det/Leg"
            index = name_text.upper().find(name_type)
            # Extract the substring starting from the index of "Leg/Det"
            name_text = name_text[index:]
    
    return name_text

def findAndFormatLeg(text):
    #Find first match for "Leg" and then filters out the next characters until an UPPER CHARACTER is reached (beginning of name)
    match = re.search(r'Leg[^A-Z]*(?P<first_upper>[A-Z]).*', text)
    if match:
        extracted_text = "Leg: " + match.group('first_upper') + re.sub(r'\s+', ' ', match.group()[(match.start('first_upper')+1):])
        return extracted_text
    else: return text

def findAndFormatDet(text):
    #Find first match for "Leg" and then filters out the next characters until an UPPER CHARACTER is reached (beginning of name)
    #Breakdown:
    #r'Leg searches for the pattern 'Leg'
    #[^A-Z] searches for any characters that are NOT uppercase - ^ is a negation * gobbles it up
    #?^<first_upper> is a capturing group which captures the first upper case letter
    #.* gobbles up all characters after the first upper case letter after 'Leg'
    #match.group('first_upper'): This retrieves the uppercase letter captured by the named group "first_upper" in the regular expression.
    #match.group()[(match.start('first_upper')+1):]: This retrieves the substring following the first uppercase letter in the matched text. (match.start('first_upper')+1) gets the index position of the character following the first uppercase letter, and match.group()[(match.start('first_upper')+1):] retrieves the substring from that index position to the end of the matched text.

    #Match for 'Leg' (uppercase L)
    match = re.search(r'Leg[^A-Z]*(?P<first_upper>[A-Z]).*', text)
    #Match for 'Det' (uppercase D)
    match2 = re.search(r'Det[^A-Z]*(?P<first_upper>[A-Z]).*', text)
    #Match for 'det' (lowercase D) 
    match3 = re.search(r'det[^A-Z]*(?P<first_upper>[A-Z]).*', text) #:OOOOO
    if match:
        extracted_text = "Det: " + match.group('first_upper') + match.group()[(match.start('first_upper')+1):]
        return extracted_text
    if match2:
        extracted_text2 = "Det: " + match2.group('first_upper') + match2.group()[(match2.start('first_upper')+1):]
        return extracted_text2
    if match3:
        extracted_text3 = "Det: " + match3.group('first_upper') + match3.group()[(match3.start('first_upper')+1):]
        return extracted_text3
    else: return text
    
def remove_digits(input_string):
    result = ''
    for char in input_string:
        if (not char.isdigit()) and ('/' != char):
            result += char

    return result

def remove_date_month(input_string):
    # Regex that removes every instance of the string "date" no matter case
    pattern = re.compile("date", re.IGNORECASE)
    input_string = pattern.sub("", input_string)
    
    # Regex that removes every instance of a month as a string no matter case
    month_list = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
    # Added word boundaries to the regex. This means that for example if a string contains a month as a substring it won't be deleted
    month = re.compile(r'\b(?:%s)\b' % '|'.join(month_list), re.IGNORECASE)

    return month.sub("", input_string)