from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load the entire model
tokenizer = AutoTokenizer.from_pretrained("model")
model = AutoModelForTokenClassification.from_pretrained("model")
model.eval()  # Set the model to evaluation mode

label_to_id = {"0": 0, "B-LEG": 1, "B-LOCATION": 2, "B-DATE": 3, "B-SPECIMEN": 4, "B-DET": 5, "B-COORD": 6, "-100": -100}

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

# Perform prediction on "hello world!"
sentence = 'Plantae groenlandica' + 'Artemisia borealis' + 'Nigerdlikasik.' + "15° 42.61' S, 61° 46.82' W" + '1805-01-10' + 'legit: Böcher, Tyge' + 'Det: Fredskild, Bent'
tokens, labels = predict_new_sentence(sentence, tokenizer, model, label_to_id)

# Display tokens and their predicted labels
#print("Tokens:", tokens)
#print("Predicted Labels:", labels)

# print specimen
# Filtering and printing the tokens corresponding to 'B-SPECIMEN' in labels
specimens = " ".join([token for label, token in zip(labels, tokens) if label == 'B-SPECIMEN'])
locations = " ".join([token for label, token in zip(labels, tokens) if label == 'B-LOCATION'])
leg = " ".join([token for label, token in zip(labels, tokens) if label == 'B-LEG'])
det = " ".join([token for label, token in zip(labels, tokens) if label == 'B-DET'])
date = " ".join([token for label, token in zip(labels, tokens) if label == 'B-DATE'])
coord = " ".join([token for label, token in zip(labels, tokens) if label == 'B-COORD'])

interests = [specimens, locations, leg, det, date, coord]
# strip hastags from each string
for i, elm in enumerate(interests):
    elm = elm.replace('#', "")
    interests[i] = elm

print("Specimen:", interests[0])
print("Locations:", interests[1])
print("Legs:", interests[2])
print("Det:", interests[3])
print("Date:", interests[4])
print("Coord:", interests[5])