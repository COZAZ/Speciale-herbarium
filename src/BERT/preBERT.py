from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModelForTokenClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from tqdm import tqdm

## Loading the model
tokenizer = AutoTokenizer.from_pretrained("Maltehb/danish-bert-botxo")
model = AutoModelForPreTraining.from_pretrained("Maltehb/danish-bert-botxo")

# Example of the dataset format
dataset = [
    {"tokens": ["hej", ",", "mit", "navn", "er", "john", "."], "labels": ["O", "O", "O", "O", "O", "B-PERSON", "O"]},
    # Add more sentences here
]

## Encoing the examples

def encode_examples(examples, label_to_id, max_length=512):
    input_ids = []
    attention_masks = []
    label_ids = []

    for example in examples:
        tokens = []
        label_ids_example = []
        for word, label in zip(example["tokens"], example["labels"]):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the first label (e.g., B-LOCATION) for the first token of the word, and padding labels (e.g., -100) for the remaining tokens
            label_ids_example.extend([label_to_id[label]] + [-100] * (len(word_tokens) - 1))
        
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        label_ids_example = [-100] + label_ids_example + [-100]
        input_ids_example = tokenizer.convert_tokens_to_ids(tokens)

        attention_mask_example = [1] * len(input_ids_example)
        
        # Padding to max_length
        padding_length = max_length - len(input_ids_example)
        input_ids_example += [0] * padding_length
        attention_mask_example += [0] * padding_length
        label_ids_example += [-100] * padding_length

        input_ids.append(input_ids_example)
        attention_masks.append(attention_mask_example)
        label_ids.append(label_ids_example)
    
    return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(label_ids)

# Example usage
label_to_id = {"O": 0, "B-PERSON": 1, "B-LOCATION": 2, "B-DATE": 3, "B-SPECIMEN": 4, "-100": -100}
input_ids, attention_masks, label_ids = encode_examples(dataset, label_to_id)

num_labels = len(label_to_id) - 1  # Subtracting one because -100 is not a real label but a padding token
model = AutoModelForTokenClassification.from_pretrained("Maltehb/danish-bert-botxo", num_labels=num_labels)

# Create TensorDataset and DataLoader
dataset = TensorDataset(input_ids, attention_masks, label_ids)
dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=8)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

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
