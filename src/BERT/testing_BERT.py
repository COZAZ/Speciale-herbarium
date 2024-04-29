import json
import numpy as np
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

    # Used to store what cases BERT struggles with
    hardcases = []
    easycases = []

    # List to hold TP [0], FP [1] and FN [2] values
    specimen_comparisons = [0,0,0]
    location_comparisons = [0,0,0]
    leg_comparisons = [0,0,0]
    det_comparisons = [0,0,0]
    date_comparisons = [0,0,0]
    coord_comparisons = [0,0,0]

    # Score containers:
    # [0] -> Precision
    # [1] -> Recall
    # [2] -> F1
    specimen_scores = [0,0,0]
    location_scores = [0,0,0]
    leg_scores = [0,0,0]
    det_scores = [0,0,0]
    date_scores = [0,0,0]
    coord_scores = [0,0,0]
    
    for i in range(data_points):
        current_true_text = trueText[i]
        current_pred_text = predText[i]
    
        for elm in label_score:
            current_class = elm[0]

            true_token = extract_token_true(current_true_text, current_class).replace(" ", "")
            pred_token = extract_token_pred(current_pred_text, current_class).replace(" ", "")

            current_similarity = fuzz.ratio(true_token, pred_token) / 100 # Comparing strings

            if current_similarity < 0.3 and len(hardcases) <= 20:
                hardcases.append(("Current Class: {0}".format(current_class), "True Token: {0}".format(true_token), "Pred Token: {0}".format(pred_token)))
            if current_similarity > 0.8 and len(easycases) <= 20:
                easycases.append(("Current Class: {0}".format(current_class), "True Token: {0}".format(true_token), "Pred Token: {0}".format(pred_token)))
            elm[1] += current_similarity

            # Counting specimens
            if current_class == "B-SPECIMEN":
                specimen_total += 1

            if current_class == "B-SPECIMEN" and current_similarity >= 0.75:
                correct_specimens_low += 1
                specimen_comparisons[0] += 1 # TP

            elif current_class == "B-SPECIMEN" and current_similarity < 0.75:
                specimen_comparisons = countFPFN(specimen_comparisons, current_true_text, pred_token, "B-SPECIMEN")

            if current_class == "B-SPECIMEN" and current_similarity >= 0.9:
                correct_specimens_high += 1
            
            # Counting Locations
            if current_class == "B-LOCATION":
                location_total += 1

            if current_class == "B-LOCATION" and current_similarity >= 0.75:
                correct_locations_low += 1
                location_comparisons[0] += 1 # TP

            elif current_class == "B-LOCATION" and current_similarity < 0.75:
                location_comparisons = countFPFN(location_comparisons, current_true_text, pred_token, "B-LOCATION")

            if current_class == "B-LOCATION" and current_similarity >= 0.9:
                correct_locations_high += 1
            
            # Counting leg
            if current_class == "B-LEG":
                leg_total += 1

            if current_class == "B-LEG" and current_similarity >= 0.75:
                correct_legs_low += 1
                leg_comparisons[0] += 1 # TP
            
            elif current_class == "B-LEG" and current_similarity < 0.75:
                leg_comparisons = countFPFN(leg_comparisons, current_true_text, pred_token, "B-LEG")

            if current_class == "B-LEG" and current_similarity >= 0.9:
                correct_legs_high += 1
            
            # Counting det
            if current_class == "B-DET":
                det_total += 1

            if current_class == "B-DET" and current_similarity >= 0.75:
                correct_dets_low += 1
                det_comparisons[0] += 1 # TP
            
            elif current_class == "B-DET" and current_similarity < 0.75:
                det_comparisons = countFPFN(det_comparisons, current_true_text, pred_token, "B-DET")

            if current_class == "B-DET" and current_similarity >= 0.9:
                correct_dets_high += 1
            
            # Counting date
            if current_class == "B-DATE":
                date_total += 1

            if current_class == "B-DATE" and current_similarity >= 0.75:
                correct_dates_low += 1
                date_comparisons[0] += 1 # TP
            
            elif current_class == "B-DATE" and current_similarity < 0.75:
                date_comparisons = countFPFN(date_comparisons, current_true_text, pred_token, "B-DATE")

            if current_class == "B-DATE" and current_similarity >= 0.9:
                correct_dates_high += 1

            # Counting coords
            if current_class == "B-COORD":
                coord_total += 1

            if current_class == "B-COORD" and current_similarity >= 0.75:
                correct_coords_low += 1
                coord_comparisons[0] += 1 # TP
            
            elif current_class == "B-COORD" and current_similarity < 0.75:
                coord_comparisons = countFPFN(coord_comparisons, current_true_text, pred_token, "B-COORD")

            if current_class == "B-COORD" and current_similarity >= 0.9:
                correct_coords_high += 1

    overall_score = 0
    for elm in label_score:
        elm[1] = round((elm[1] / data_points) * 100, 2)
        overall_score += elm[1]
        
    overall_score = round(overall_score / len(label_score), 2)

    # Precision
    specimen_scores[0] = specimen_comparisons[0] / (specimen_comparisons[0] + specimen_comparisons[1])
    location_scores[0] = location_comparisons[0] / (location_comparisons[0] + location_comparisons[1])
    leg_scores[0] = leg_comparisons[0] / (leg_comparisons[0] + leg_comparisons[1])
    det_scores[0] = det_comparisons[0] / (det_comparisons[0] + det_comparisons[1])
    date_scores[0] = date_comparisons[0] / (date_comparisons[0] + date_comparisons[1])
    coord_scores[0] = coord_comparisons[0] / (coord_comparisons[0] + coord_comparisons[1])

    # Recall
    specimen_scores[1] = specimen_comparisons[0] / (specimen_comparisons[0] + specimen_comparisons[2])
    location_scores[1] = location_comparisons[0] / (location_comparisons[0] + location_comparisons[2])
    leg_scores[1] = leg_comparisons[0] / (leg_comparisons[0] + leg_comparisons[2])
    det_scores[1] = det_comparisons[0] / (det_comparisons[0] + det_comparisons[2])
    date_scores[1] = date_comparisons[0] / (date_comparisons[0] + date_comparisons[2])
    coord_scores[1] = coord_comparisons[0] / (coord_comparisons[0] + coord_comparisons[2])

    # F1
    specimen_scores[2] = 2 * ((specimen_scores[0] * specimen_scores[1]) / (specimen_scores[0] + specimen_scores[1]))
    location_scores[2] = 2 * ((location_scores[0] * location_scores[1]) / (location_scores[0] + location_scores[1]))
    leg_scores[2] = 2 * ((leg_scores[0] * leg_scores[1]) / (leg_scores[0] + leg_scores[1]))
    det_scores[2] = 2 * ((det_scores[0] * det_scores[1]) / (det_scores[0] + det_scores[1]))
    date_scores[2] = 2 * ((date_scores[0] * date_scores[1]) / (date_scores[0] + date_scores[1]))
    coord_scores[2] = 2 * ((coord_scores[0] * coord_scores[1]) / (coord_scores[0] + coord_scores[1]))

    average_precision = np.mean([specimen_scores[0], location_scores[0], leg_scores[0], det_scores[0], date_scores[0], coord_scores[0]])
    average_recall = np.mean([specimen_scores[1], location_scores[1], leg_scores[1], det_scores[1], date_scores[1], coord_scores[1]])
    average_f1 = np.mean([specimen_scores[2], location_scores[2], leg_scores[2], det_scores[2], date_scores[2], coord_scores[2]])
  
    print("Precsion scores:")
    print("SPECIMEN: {0}%".format(round(specimen_scores[0]*100, 2)))
    print("LOCATION: {0}%".format(round(location_scores[0]*100, 2)))
    print("LEG: {0}%".format(round(leg_scores[0]*100, 2)))
    print("DET: {0}%".format(round(det_scores[0]*100, 2)))
    print("DATE: {0}%".format(round(date_scores[0]*100, 2)))
    print("COORD: {0}%".format(round(coord_scores[0]*100, 2)))
    print("Average model precision: {0}%".format(round(average_precision*100, 2)))
   
    print("\nRecall scores:")
    print("SPECIMEN: {0}%".format(round(specimen_scores[1]*100, 2)))
    print("LOCATION: {0}%".format(round(location_scores[1]*100, 2)))
    print("LEG: {0}%".format(round(leg_scores[1]*100, 2)))
    print("DET: {0}%".format(round(det_scores[1]*100, 2)))
    print("DATE: {0}%".format(round(date_scores[1]*100, 2)))
    print("COORD: {0}%".format(round(coord_scores[1]*100, 2)))
    print("Average model recall: {0}%".format(round(average_recall*100, 2)))

    print("\nF1 scores:")
    print("SPECIMEN: {0}%".format(round(specimen_scores[2]*100, 2)))
    print("LOCATION: {0}%".format(round(location_scores[2]*100, 2)))
    print("LEG: {0}%".format(round(leg_scores[2]*100, 2)))
    print("DET: {0}%".format(round(det_scores[2]*100, 2)))
    print("DATE: {0}%".format(round(date_scores[2]*100, 2)))
    print("COORD: {0}%".format(round(coord_scores[2]*100, 2)))
    print("Average model F1: {0}%\n".format(round(average_f1*100, 2)))

    return label_score, overall_score, (correct_specimens_high, correct_specimens_low, specimen_total), (correct_locations_high, correct_locations_low, location_total), (correct_legs_high, correct_legs_low, leg_total), (correct_dets_high, correct_dets_low, det_total), (correct_dates_high, correct_dates_low, date_total), (correct_coords_high, correct_coords_low, coord_total), hardcases, easycases 

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

def examineBadPredictions(t,p):
    t0 = extract_token_true(t, "B-SPECIMEN").replace(" ", "")
    t1 = extract_token_true(t, "B-LOCATION").replace(" ", "")
    t2 = extract_token_true(t, "B-LEG").replace(" ", "")
    t3 = extract_token_true(t, "B-DET").replace(" ", "")
    t4 = extract_token_true(t, "B-DATE").replace(" ", "")
    t5 = extract_token_true(t, "B-COORD").replace(" ", "")

    s0 = (p, t0, "B-SPECIMEN", fuzz.ratio(t0, p) / 100)
    s1 = (p, t1, "B-LOCATION", fuzz.ratio(t1, p) / 100)
    s2 = (p, t2, "B-LEG", fuzz.ratio(t2, p) / 100)
    s3 = (p, t3, "B-DET", fuzz.ratio(t3, p) / 100)
    s4 = (p, t4, "B-DATE", fuzz.ratio(t4, p) / 100)
    s5 = (p, t5, "B-COORD", fuzz.ratio(t5, p) / 100)

    ss = [s0,s1,s2,s3,s4,s5]

    largest_s = max(ss, key=lambda x: x[3])

    print(largest_s)
    return largest_s

def countFPFN(comps, t, p, type):
    comps[2] += 1 # FN
    other_match = examineBadPredictions(t, p)
    if other_match[2] != type and other_match[3] >= 0.75:
        comps[1] += 1 # FP
    
    return comps