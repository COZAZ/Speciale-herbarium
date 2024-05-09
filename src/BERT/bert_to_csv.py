import json
import csv
import unicodedata
import re
import os
import numpy as np
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

    data_slice1 = data[1:]
    sorted_data1 = sorted(data_slice1, key=lambda x: int(x[0]))

    unique_BERT_data = removeDuplicates(sorted_data1)

    with open("../herbarium_BERT.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(data[0])
        for row in unique_BERT_data:
            clean_row = []
            for string in row:
                current_clean_string = unicodedata.normalize("NFKD", string).encode("ascii", "replace").decode()
                clean_row.append(current_clean_string)
            csv_writer.writerow(clean_row)

    print("CSV without postprocessing created.")

    with open('ocr_post.json', 'r') as f:
        ocr_text = json.load(f)

    for i, elm in enumerate(predicted):
        threshold = 75
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
            spec_calc = fuzz.partial_ratio(text_piece, pred_spec)
            if spec_calc > spec_score:
                spec_score = spec_calc
                spec_correct_index = j
            
            loc_calc = fuzz.partial_ratio(text_piece, pred_loc)
            if loc_calc > loc_score:
                loc_score = loc_calc
                loc_correct_index = j

            leg_calc = fuzz.partial_ratio(text_piece, pred_leg)
            if leg_calc > leg_score:
                leg_score = leg_calc
                leg_correct_index = j

            det_calc = fuzz.partial_ratio(text_piece, pred_det)
            if det_calc > det_score:
                det_score = det_calc
                det_correct_index = j
            # If Det never found in OCR reading
            if pred_det == "":
                det_score = 101

            date_calc = fuzz.partial_ratio(text_piece, pred_date)
            if date_calc > date_score:
                date_score = date_calc
                date_correct_index = j

            coord_calc = fuzz.partial_ratio(text_piece, pred_coord)
            if coord_calc > coord_score:
                coord_score = coord_calc
                coord_correct_index = j

        if (spec_score >= threshold):
            spec_csv = ocr_object[spec_correct_index]

            spec_csv = cleanSpeciesName(spec_csv)

            data[i+1][1] = spec_csv
            
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

            key_words = ["lat", "long"]
            for elm in key_words:
                if (loc_csv.lower().find(elm) != -1):
                    elm_index = loc_csv.lower().find(elm)
                    loc_csv = loc_csv[:elm_index]

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

            leg_index = date_csv.upper().find("LEG")
            det_index = date_csv.upper().find("DET")

            if leg_index != -1:
                date_csv = date_csv[:leg_index]
            elif det_index != -1:
                date_csv = date_csv[:det_index]
            
            date_csv = clean_dates(date_csv)
            date_csv = ' '.join(date_csv.split())
            
            data[i+1][5] = date_csv

        if coord_score >= threshold:
            coord_csv = ocr_object[coord_correct_index]
            
            if "LAT" in coord_csv.upper():
                coord_index = coord_csv.upper().find("LAT")
                coord_csv = coord_csv[coord_index:]
            
            coord_csv = ' '.join(coord_csv.split())
            coord_csv = removeCoordFiller(coord_csv)
            coord_csv = addMissingWords(coord_csv)
            
            data[i+1][6] = coord_csv        

    data_slice2 = data[1:]
    sorted_data2 = sorted(data_slice2, key=lambda x: int(x[0]))
    
    unique_post_data = removeDuplicates(sorted_data2)

    with open("../herbarium_post.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(data[0])
        for row in unique_post_data:
            clean_row = []
            for string in row:
                current_clean_string = unicodedata.normalize("NFKD", string).encode("ascii", "replace").decode()
                clean_row.append(current_clean_string)
            csv_writer.writerow(clean_row)

    print("CSV with postprocessing created")

def cleanSpeciesName(species_text):
    key_words = ["hab", "loc", "date", "alt", "leg", "det", "lat", "long"]
    for elm in key_words:

        if (species_text.lower().find(elm + " ") != -1) or (species_text.lower().find(elm + ".") != -1) or (species_text.lower().find(elm + ":") != -1) or (species_text.lower().find(elm + ",") != -1):
            elm_index = species_text.lower().find(elm)
            species_text = species_text[:elm_index]

    if len(Convert(species_text)) >= 10:
        species_text = ' '.join(Convert(species_text)[:10])
    
    return species_text

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

def Convert(string): 
    li = list(string.split(" ")) 
    return li 
    
def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)
    
def has_letters(inputString):
    return re.search('[a-zA-Z]', inputString)
    
def contains_only_numbers_and_commas(input_string):
    pattern = r'^[0-9,/.\-]+$'
    return bool(re.match(pattern, input_string))
    
def clean_dates(input_string):
    month_list = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "date"]
    input_list = Convert(input_string)
    elms_to_pop = []
    for i, elm in enumerate(input_list):
        if elm.lower() in month_list:
            continue
        if has_letters(input_list[i]):
            elms_to_pop.append(i)
        if sum(c.isdigit() for c in elm) < 4:
            if input_list[i-1].lower() not in month_list:
                elms_to_pop.append(i)
        if sum(c.isdigit() for c in elm) >= 4:
            if not contains_only_numbers_and_commas(elm):
                elms_to_pop.append(i)
            
    output_list = np.delete(input_list,elms_to_pop)
    retval = " ".join(output_list)

    return retval

def removeCoordFiller(input):
    coords_no_words = removeWords(input)
    coords_no_words = coords_no_words.split()

    WE_occurences = []
    for i, w in enumerate(coords_no_words):
        if ('W' in w) or ('E' in w):
            WE_occurences.append(i)
    
    if WE_occurences != []:
        last_WE = WE_occurences[-1]
        coords_no_words = coords_no_words[:last_WE+1]
        
    res = ' '.join(coords_no_words)

    return res

def countAlpha(input):
    c = 0
    for char in input:
        if char.isalpha():
            c += 1

    return c

def removeWords(input):
    words = input.split()
    keep_words = []
    for w in words:
        if ("LAT" not in w.upper()) and ("LONG" not in w.upper()):
            alpha_count = countAlpha(w)
            
            if (alpha_count < 2):
                keep_words.append(w)
        else:
            keep_words.append(w)
    
    keep_words = ' '.join(keep_words)

    return keep_words

def addMissingWords(input):
    coord = input
    if coord[0].upper() != 'L':
        coord = "Lat " + coord
    
    if "LONG" not in coord.upper():
        coord = coord.split()
        new_coord = []
        for w in coord:
            if (('N' in w) or ('S' in w)):
                    new_coord.append(w)
                    new_coord.append("Long")
            else:
                if not ((len(w) == 1) and (w in [',','\'',':','.'])):
                    new_coord.append(w)

        coord = ' '.join(new_coord)
    
    return coord

def removeDuplicates(data):
    # We keep only one data entry per catalog number
    # since images with multiple institutional/annotation labels all describe the same plant
    unique_data = []
    encountered_catalog_numbers = []
    for entry in data:
        catalog_number = entry[0]
        if catalog_number not in encountered_catalog_numbers:
            unique_data.append(entry)
            encountered_catalog_numbers.append(catalog_number)
    
    return unique_data