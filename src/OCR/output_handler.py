import json

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
