import json
from difflib import SequenceMatcher
from BERT.pred import parse_ocr_text
from thefuzz import fuzz

# BERT model accuracy
def testBERTAccuracy(data_points):
    with open("../synth_text_data_test2.json", 'r') as f:
        test_text = json.load(f)

    trueText = test_text
    trueText_JSONized = []

    # Convert text to (stylized)JSON format
    counter = 1
    for obj in trueText:
        obj_text = obj["tokens"]
        obj_json = {"image": str(counter) + ".jpg", "label_type": "X", "text": obj_text}

        trueText_JSONized.append(obj_json)
        counter += 1
                    
    predText = parse_ocr_text(trueText_JSONized, True)
    label_score = [["B-SPECIMEN", 0], ["B-LOCATION", 0], ["B-LEG", 0], ["B-DET", 0], ["B-DATE", 0], ["B-COORD", 0]]

    correct_specimens_low = 0
    correct_specimens_high = 0
    specimen_total = 0

    correct_locations_low = 0
    correct_locations_high = 0
    location_total = 0

    correct_legs_low = 0
    correct_legs_high = 0
    leg_total = 0

    correct_dets_low = 0
    correct_dets_high = 0
    det_total = 0

    correct_dates_low = 0
    correct_dates_high = 0
    date_total = 0

    correct_coords_low = 0
    correct_coords_high = 0
    coord_total = 0

    for i in range(data_points):
        current_true_text = trueText[i]
        current_pred_text = predText[i]
    
        for elm in label_score:
            current_class = elm[0]

            true_token = extract_token_true(current_true_text, current_class).replace(" ", "")
            pred_token = extract_token_pred(current_pred_text, current_class).replace(" ", "")

            #current_similarity = SequenceMatcher(None, true_token, pred_token).ratio() # Comparing strings
            current_similarity = fuzz.ratio(true_token, pred_token) / 100 # Comparing strings
            elm[1] += current_similarity

            # Counting specimens
            if current_class == "B-SPECIMEN":
                specimen_total += 1

            if current_class == "B-SPECIMEN" and current_similarity >= 0.75:
                correct_specimens_low += 1

            if current_class == "B-SPECIMEN" and current_similarity >= 0.9:
                correct_specimens_high += 1
            
            # Counting Locations
            if current_class == "B-LOCATION":
                location_total += 1

            if current_class == "B-LOCATION" and current_similarity >= 0.75:
                correct_locations_low += 1

            if current_class == "B-LOCATION" and current_similarity >= 0.9:
                correct_locations_high += 1
            
            # Counting leg
            if current_class == "B-LEG":
                leg_total += 1

            if current_class == "B-LEG" and current_similarity >= 0.75:
                correct_legs_low += 1

            if current_class == "B-LEG" and current_similarity >= 0.9:
                correct_legs_high += 1
            
            # Counting det
            if current_class == "B-DET":
                det_total += 1

            if current_class == "B-DET" and current_similarity >= 0.75:
                correct_dets_low += 1

            if current_class == "B-DET" and current_similarity >= 0.9:
                correct_dets_high += 1
            
            # Counting date
            if current_class == "B-DATE":
                date_total += 1

            if current_class == "B-DATE" and current_similarity >= 0.75:
                correct_dates_low += 1

            if current_class == "B-DATE" and current_similarity >= 0.9:
                correct_dates_high += 1

            # Counting coords
            if current_class == "B-COORD":
                coord_total += 1

            if current_class == "B-COORD" and current_similarity >= 0.75:
                correct_coords_low += 1

            if current_class == "B-COORD" and current_similarity >= 0.9:
                correct_coords_high += 1

    overall_score = 0
    for elm in label_score:
        elm[1] = round((elm[1] / data_points) * 100, 2)
        overall_score += elm[1]
        
    overall_score = round(overall_score / len(label_score), 2)

    return label_score, overall_score, (correct_specimens_high, correct_specimens_low, specimen_total), (correct_locations_high, correct_locations_low, location_total), (correct_legs_high, correct_legs_low, leg_total), (correct_dets_high, correct_dets_low, det_total), (correct_dates_high, correct_dates_low, date_total), (correct_coords_high, correct_coords_low, coord_total) 

def extract_token_true(text, label_type):
    # Find index of 'B-label_type' in labels, where label_type could be 'SPECIMEN', 'Date' etc.
    label_indices = [index for index, label in enumerate(text["labels"]) if label == label_type]
    
    label_token = ""

    for index in label_indices:
        label_token += text["tokens"][index]
        
    return label_token

def extract_token_pred(text, label_type):
    for elm in text:
        if elm[0] == label_type:
            return elm[1]