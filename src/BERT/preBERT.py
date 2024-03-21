from transformers import BertTokenizer, BertForTokenClassification
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from tqdm import tqdm
import json

# Example of the dataset format
"""
dataset = [
    {"tokens": ["hej", ",", "mit", "navn", "er", "john", "."], "labels": ["O", "O", "O", "O", "O", "B-PERSON", "O"]},
    # Add more sentences here
    # Image: 697718.jpg    
    # TODO: TIL BULUT
    {"tokens": ["G.B.U.", "508"], "labels": ["O", "O"]},
    {"tokens": ["Museum", "Botanicum", "Hauniense"], "labels": ["O", "O", "O"]},
    {"tokens": ["Grønlands", "Botaniske", "Undersøgelse", "Plantae", "Groenlandicae"], "labels": ["O", "O", "O", "O", "O"]},
    {"tokens": ["E.3."], "labels": ["O"]},
    {"tokens": ["Woodsia", "ilvensis", "(L.)", "R.", "Br."], "labels": ["B-SPECIMEN", "B-SPECIMEN", "B-SPECIMEN", "B-SPECIMEN", "B-SPECIMEN"]},
    {"tokens": ["Angmagssalik", "distr:", "Sieraq"], "labels": ["B-LOCATION", "B-LOCATION", "B-LOCATION"]},
    {"tokens": ["65°", "56'", "N.", "lat.", "37°", "09'", "W.", "long."], "labels": ["O", "O", "O", "O", "O", "O", "O", "O"]},
    {"tokens": ["27.", "Juli", "1970"], "labels": ["B-DATE", "B-DATE", "B-DATE"]},
    {"tokens": ["Leg.:", "Mette", "Astrup", "Lars", "Kliim", "Nielsen"], "labels": ["O", "B-PERSON", "B-PERSON", "I-PERSON", "I-PERSON", "I-PERSON"]},
    {"tokens": ["Det.:"], "labels": ["O"]}
]
"""

## Encoding the examples
def encode_examples(examples, label_to_id, max_length=512):
    input_ids = []
    attention_masks = []
    label_ids = []

    ## Loading the model
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-multilingual-uncased")
    #model = BertForTokenClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=len(label_to_id))

    for example in examples:
        tokens = ["[CLS]"]
        label_ids_example = [-100]  # Using -100 for [CLS] to ignore it in loss calculation

        for word, label in zip(example["tokens"], example["labels"]):
            word_tokens = tokenizer.tokenize(word)
            # Check if adding this word exceeds the max_length when including [SEP] token
            if len(tokens) + len(word_tokens) + 1 > max_length:
                break  # Stop adding tokens for this example if max_length is reached

            tokens.extend(word_tokens)
            # Use the first label (e.g., B-LOCATION) for the first token of the word, and padding labels (e.g., -100) for the remaining tokens
            label_ids_example.extend([label_to_id[label]] + [-100] * (len(word_tokens) - 1))

        tokens.append("[SEP]")
        label_ids_example.append(-100)  # Using -100 for [SEP] to ignore it in loss calculation

        input_ids_example = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask_example = [1] * len(input_ids_example)
        
        # Note: No need to pad here as we ensure the length does not exceed max_length
        input_ids.append(input_ids_example)
        attention_masks.append(attention_mask_example)
        label_ids.append(label_ids_example)
    
    # Padding moved here to ensure all sequences are padded to the same length
    input_ids_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=0)
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(mask) for mask in attention_masks], batch_first=True, padding_value=0)
    label_ids_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(ids) for ids in label_ids], batch_first=True, padding_value=-100)

    return input_ids_padded, attention_masks_padded, label_ids_padded, tokenizer

def train_bert():
    label_to_id = {"0": 0, "B-LEG": 1, "B-LOCATION": 2, "B-DATE": 3, "B-SPECIMEN": 4, "B-DET": 5, "B-COORD": 6, "-100": -100}

    # load JSON data
    f = open('synth_data.json')
    dataset = json.load(f)

    # Example usage
    input_ids, attention_masks, label_ids, tokenizer = encode_examples(dataset, label_to_id)

    num_labels = len(label_to_id) - 1  # Subtracting one because -100 is not a real label but a padding token
    model = BertForTokenClassification.from_pretrained("google-bert/bert-base-multilingual-uncased", num_labels=num_labels)

    # Create TensorDataset and DataLoader
    dataset = TensorDataset(input_ids, attention_masks, label_ids)
    dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=8)

    # Optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-05)

    model.train()
    for epoch in range(3):  # for a few epochs
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Training Epoch {epoch}"):
            batch = tuple(t.to(model.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
            
            model.zero_grad()
            
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            total_loss += loss.item()
            
            optimizer.step()
        
        print(f"Average loss: {total_loss / len(dataloader)}")

    tokenizer.save_pretrained('BERT_model')
    model.save_pretrained('BERT_model')