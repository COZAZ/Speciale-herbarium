import json
from difflib import SequenceMatcher
from numpy import mean

# Save the OCR output text as a JSON file
def save_ocr_output(text_data):
    dict_list = [tuple_to_dict(t) for t in text_data]

    synthJsonData = json.dumps(dict_list, indent=4)
    with open("../ocr_output.json", 'w') as json_file:
        json_file.seek(0)
        json_file.truncate()
        json_file.write(synthJsonData)

    print("OCR output text saved")

# Convert tuples to dictionaries
def tuple_to_dict(t):
    return {"image": t[0], "label_type": t[1], "text": t[2]}

# Evaluate the performance of the OCR model using string similarity
def evaluate_craft_ocr():
    predicted_texts = None
    true_texts = None

    with open("../ocr_output.json", 'r') as f:
        predicted_texts = json.load(f)
    with open("../ocr_true_text.json", 'r') as f:
        true_texts = json.load(f)
        
    similarity_scores = []

    for entry_pred in predicted_texts:
        current_highest_similarity = 0
        full_pred_text = " ".join(entry_pred["text"])
        
        for entry_true in true_texts:
            # Manuel added this line by write as one line
            full_true_text = entry_true["text"]

            if entry_true["image"] == entry_pred["image"]:
                new_similarity = SequenceMatcher(None, full_pred_text, full_true_text).ratio()

                if new_similarity > current_highest_similarity:
                    current_highest_similarity = new_similarity
                
        similarity_scores.append(current_highest_similarity)
    
    total_similarity_score = mean(similarity_scores) * 100
    
    return round(total_similarity_score, 2)