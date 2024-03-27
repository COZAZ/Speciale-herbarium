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
                    
    predText_validation = parse_ocr_text(trueText_JSONized, True)
    label_score = [["B-SPECIMEN", 0], ["B-LOCATION", 0], ["B-LEG", 0], ["B-DET", 0], ["B-DATE", 0], ["B-COORD", 0]]

    for i in range(data_points):
        current_true_text = trueText_validation[i]
        current_pred_text = predText_validation[i]

        # TODO: Ask Kim how we should compare the different labels
        for elm in label_score:
            true_token = extract_token_true(current_true_text, elm[0])
            pred_token = extract_token_pred(current_pred_text, elm[0])
            current_similarity = SequenceMatcher(None, true_token, pred_token).ratio()
            elm[1] += current_similarity
    
    for elm in label_score:
        elm[1] = round((elm[1] / data_points) * 100, 2)

    return label_score

def extract_token_true(text, label_type):
    # Find index of 'B-label_type' in labels, where label_type could be 'SPECIMEN', 'Date' etc.
    label_index = text["labels"].index(label_type)
    # Find the corresponding token
    label_token = text["tokens"][label_index]

    return label_token

def extract_token_pred(text, label_type):
    for elm in text:
        if elm[0] == label_type:
            return elm[1]