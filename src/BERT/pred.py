from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModelForTokenClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from tqdm import tqdm
import json

# Load the entire model
tokenizer = AutoTokenizer.from_pretrained("Maltehb/danish-bert-botxo")
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
sentence = "Det. Bobathan Bobathansen"
tokens, labels = predict_new_sentence(sentence, tokenizer, model, label_to_id)

# Display tokens and their predicted labels
print("Tokens:", tokens)
print("Predicted Labels:", labels)