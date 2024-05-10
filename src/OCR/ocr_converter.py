import json

def AdjustTextLayout():
    with open("ocr_coords.json", 'r') as f1:
        ocr_text = json.load(f1)

    sorted_ocr = []
    for i, elm in enumerate(ocr_text):
        sorted_data = sorted(elm["text"], key=lambda x: x[0][0][0])
        sorted_ocr.append(sorted_data)
        ocr_text[i]["text"] = sorted_data

    s_data = json.dumps(ocr_text, indent=4)
    with open('ocr_sorted.json', 'w') as f2:
        f2.seek(0)
        f2.truncate()
        f2.write(s_data)

    complete_new_lines = []
    for i, elm in enumerate(ocr_text):
        new_lines = []
        text_ocr = ocr_text[i]["text"]
        for j in range(len(text_ocr)):
            set_continue = False
            for elm in new_lines:
                if text_ocr[j][1] in elm:
                    set_continue = True
            if set_continue:
                continue
            curr_line = text_ocr[j][1]
            curr_right_x = text_ocr[j][0][1][0]
            curr_top_y = text_ocr[j][0][0][1]
            curr_bottom_y = text_ocr[j][0][3][1]
            for k in range(j+1, len(text_ocr)):
                new_left_x = text_ocr[k][0][0][0]
                new_top_y = text_ocr[k][0][0][1]
                new_bottom_y = text_ocr[k][0][3][1]
                new_text = text_ocr[k][1]
                if (abs(curr_top_y - new_top_y) < 25) and (abs(curr_bottom_y - new_bottom_y) < 25):
                        if (curr_right_x < new_left_x):
                            curr_line = curr_line + " " + new_text
                        else:
                            curr_line = new_text + " " + curr_line
            new_lines.append(curr_line)
            curr_line = ""
        complete_new_lines.append(new_lines)

    for num, elm in enumerate(ocr_text):
        elm["text"] = complete_new_lines[num]

    # Open the same JSON file for writing (this will overwrite the existing content)
    synthJsonData = json.dumps(ocr_text, indent=4)
    with open('ocr_post.json', 'w') as file:
        file.seek(0)
        file.truncate()
        file.write(synthJsonData)

def removeCoords():
    with open("ocr_coords.json", 'r') as f3:
        ocr_coords = json.load(f3)
    
    modified_entries = []

    for entry in ocr_coords:
        text_elements = [text[1] for text in entry['text']]

        modified_entry = {
            'image': entry['image'],
            'label_type': entry['label_type'],
            'text': text_elements
        }

        modified_entries.append(modified_entry)

    with open('ocr_predict.json', 'w') as f4:
        json.dump(modified_entries, f4, indent=4)