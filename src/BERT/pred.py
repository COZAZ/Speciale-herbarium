from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

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

    return tokens, labels

def parse_ocr_text():
    # Load the model
    tokenizer = AutoTokenizer.from_pretrained("BERT_model")
    model = AutoModelForTokenClassification.from_pretrained("BERT_model")
    model.eval()  # Set the model to evaluation mode

    label_to_id = {"0": 0, "B-LEG": 1, "B-LOCATION": 2, "B-DATE": 3, "B-SPECIMEN": 4, "B-DET": 5, "B-COORD": 6, "-100": -100}

    # TODO: at some point, load ocr output JSON as text data
    # TODO: Ask Kim if two (or more) entries for same image in .csv file is okay (institutional and annotation labels)
    object1 = {
        "image": "999999.jpg",
        "label_type": "i",
        "text": [
            "Plants of North East Greenland E.6 Collected on the British Arcturus Expedition to Krumme Langs\u00f8, Ole R\u00f8mers Land. 2004",
            "Ranunculus pedatifidus Sm.",
            "On slope below K6 cave 29 above north side of Kumme Langs\u00f8. Lat. 740 02' N. Long 230 36' W. Alt. 250m. 21.7.04",
            "Leg RWMCorner"
        ]
    }

    object2 = {
        "image": "111111.jpg",
        "label_type": "i",
        "text": [
            "Plants of North East Greenland E.6 Collected on the British Arcturus Expedition to Krumme Langs\u00f8, Ole R\u00f8mers Land. 2004",
            "Ranunculus pedatifidus Sm.",
            "On slope below K6 cave 29 above north side of Kumme Langs\u00f8. Lat. 740 02' N. Long 230 36' W. Alt. 250m. 21.7.04",
            "Leg RWMCorner"
        ]
    }

    ocr_text_objects = [object1, object2]

    parsed_text = []
    
    for obj in ocr_text_objects:
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
        
        interests = [image_name[:-4], specimens, locations, leg, det, date, coord]
        
        # strip hastags from each string
        for i, elm in enumerate(interests):
            elm = elm.replace('#', "")
            interests[i] = elm

        parsed_text.append(interests)
        
        # Display the extracted information
        #print("Specimen:", interests[0])
        #print("Locations:", interests[1])
        #print("Legs:", interests[2])
        #print("Det:", interests[3])
        #print("Date:", interests[4])
        #print("Coord:", interests[5])
    
    return parsed_text