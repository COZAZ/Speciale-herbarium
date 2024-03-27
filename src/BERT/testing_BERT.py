from difflib import SequenceMatcher
from BERT.text_data_synthesizer import synthesize_text_data
from BERT.pred import parse_ocr_text

# BERT model accuracy
def testBERTAccuracy(data_points):
    trueText_validation = synthesize_text_data(data_points, asJson=False)
    trueText_JSONized = []

    # Convert text to (stylized)JSON format
    counter = 1
    for obj in trueText_validation:
        obj_text = obj["tokens"]
        obj_json = {"image": str(counter) + ".jpg", "label_type": "X", "text": obj_text}

        trueText_JSONized.append(obj_json)
        counter += 1
    
    # Index values info:
    # 0 - Image name
    # 1 - Specimen
    # 2 - Location
    # 3 - Leg
    # 4 - Det
    # 5 - Date
    # 6 - Coords
                    
    predText_validation = parse_ocr_text(trueText_JSONized, True)
    label_score = [["B-SPECIMEN", 1, 0], ["B-LOCATION", 2, 0], ["B-LEG", 3, 0], ["B-DET", 4, 0], ["B-DATE", 5, 0], ["B-COORD", 6, 0]]
    
    for i in range(data_points):
        current_true_text = trueText_validation[i]
        current_pred_text = predText_validation[i]

        # TODO: Ask Kim how we should compare the different labels
        for elm in label_score:
            true_token = extract_token(current_true_text, elm[0])
            pred_token = current_pred_text[elm[1]]

            current_similarity = SequenceMatcher(None, true_token, pred_token).ratio()
            elm[2] += current_similarity

    
    for elm in label_score:
        elm[2] = round((elm[2] / data_points) * 100, 2)

    return label_score

def extract_token(text, label_type):
    # Find index of 'B-label_type' in labels, where label_type could be 'SPECIMEN', 'Date' etc.
    label_index = text["labels"].index(label_type)
    # Find the corresponding token
    label_token = text["tokens"][label_index]

    return label_token