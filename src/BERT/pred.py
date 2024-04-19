import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from json_loader import load_json_file
#from difflib import SequenceMatcher

# Function to tokenize new sentences and perform predictions
def predict_new_sentence(sentence, tokenizer, model, label_to_id):
    # Tokenize the sentence
    tokens = tokenizer.tokenize(sentence)
    # Add the special tokens
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    # Convert tokens to input IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Create attention masks
    attention_mask = [1] * len(input_ids)
    # Pad sequences if necessary (here, it might not be, but included for completeness)
    input_ids = torch.tensor([input_ids])  # Add batch dimension
    attention_mask = torch.tensor([attention_mask])  # Add batch dimension
    # Perform prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    # Convert logits to label IDs
    label_ids = torch.argmax(logits, dim=2)
    # Convert label IDs to labels
    id_to_label = {value: key for key, value in label_to_id.items()}
    labels = [id_to_label[label_id.item()] for label_id in label_ids[0]]
    
    strings_remove = ["[CLS]", "[SEP]"]
    for elm in strings_remove:
        if elm in tokens:
            tokens.remove(elm)
    
    return tokens, labels

def parse_ocr_text(text_to_predict=None, use_custom_text=False):
    # Load the model
    tokenizer = AutoTokenizer.from_pretrained("BERT_model")
    model = AutoModelForTokenClassification.from_pretrained("BERT_model")
    model.eval()  # Set the model to evaluation mode

    label_to_id = {"0": 0, "B-LEG": 1, "B-LOCATION": 2, "B-DATE": 3, "B-SPECIMEN": 4, "B-DET": 5, "B-COORD": 6, "-100": -100}

    # TODO: Ask Kim if two (or more) entries for same image in .csv file is okay (institutional and annotation labels).

    if (use_custom_text == True) and (text_to_predict != None):
        ocr_text_objects = text_to_predict
    else:
        json_path = "ocr_output.json"
        ocr_text_objects = load_json_file(json_path)

    parsed_text = []
    
    #temp_count = 0
    for obj in ocr_text_objects:
        #if temp_count > 10:
        #    break
        text_string = " ".join(obj["text"])
        
        image_name = obj["image"]

        tokens, labels = predict_new_sentence(text_string, tokenizer, model, label_to_id)
        
        # Filtering and printing the tokens corresponding to 'B-SPECIMEN' in labels
        specimens = " ".join([token for label, token in zip(labels, tokens) if label == 'B-SPECIMEN'])
        locations = " ".join([token for label, token in zip(labels, tokens) if label == 'B-LOCATION'])
        leg = " ".join([token for label, token in zip(labels, tokens) if label == 'B-LEG'])
        det = " ".join([token for label, token in zip(labels, tokens) if label == 'B-DET'])
        date = " ".join([token for label, token in zip(labels, tokens) if label == 'B-DATE'])
        coord = " ".join([token for label, token in zip(labels, tokens) if label == 'B-COORD'])

        # For testing purposes
        if (use_custom_text == True) and (text_to_predict != None):
            interests = [['0', image_name[:-4]], ['B-SPECIMEN', specimens], ['B-LOCATION', locations], ['B-LEG', leg], ['B-DET', det], ['B-DATE', date], ['B-COORD', coord]]
    
            # strip hastags from each string
            for i, elm in enumerate(interests):
                elm[1] = elm[1].replace('#', "")
                
            parsed_text.append(interests)

        else:
            interests = [image_name[:-4], specimens, locations, leg, det, date, coord]
        
            # strip hastags from each string
            for i, elm in enumerate(interests):
                elm = elm.replace('#', "")
                interests[i] = elm

            parsed_text.append(interests)
        
        #temp_count += 1

    return parsed_text