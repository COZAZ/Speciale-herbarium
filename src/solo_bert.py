import json
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from tqdm import tqdm
from BERT.text_data_synthesizer import synthesize_text_data
from transformers import BertTokenizer, BertForTokenClassification

# Check if a GPU is available and select it, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

label_to_id = {"0": 0, "B-LEG": 1, "B-LOCATION": 2, "B-DATE": 3, "B-SPECIMEN": 4, "B-DET": 5, "B-COORD": 6, "-100": -100}

## Loading the model
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")

# load JSON data
f = open('synth_data.json')
dataset = json.load(f)

## Encoing the examples
def encode_examples(examples, label_to_id, max_length=512):
    input_ids = []
    attention_masks = []
    label_ids = []

    for example in examples:
        tokens = ["[CLS]"]
        label_ids_example = [-100]  # Using -100 for [CLS] to ignore it in loss calculation

        for word, label in zip(example["tokens"], example["labels"]):
            word_tokens = tokenizer.tokenize(word)
            # Check if adding this word exceeds the max_length when including [SEP] token
            if len(tokens) + len(word_tokens) + 1 > max_length:
                break  # Stop adding tokens for this example if max_length is reached

            tokens.extend(word_tokens)
            # Use the initial label for all sub-tokens of the word.
            label_ids_example.extend([label_to_id[label]] + [-100] * (len(word_tokens) - 1))

        tokens.append("[SEP]")
        label_ids_example.append(-100)

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

    return input_ids_padded, attention_masks_padded, label_ids_padded

# Example usage
input_ids, attention_masks, label_ids = encode_examples(dataset, label_to_id)

num_labels = len(label_to_id) - 1  # Subtracting one because -100 is not a real label but a padding token

# Load model from Huggingface: https://huggingface.co/google-bert/bert-base-multilingual-cased
model = BertForTokenClassification.from_pretrained("google-bert/bert-base-multilingual-cased", num_labels=num_labels)

# Move the model to the selected device (preferably gpu)
model.to(device)

# Create TensorDataset and DataLoader
dataset = TensorDataset(input_ids, attention_masks, label_ids)
dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=32)

testset = synthesize_text_data(1000, asJson=False)
input_ids, attention_masks, label_ids = encode_examples(testset, label_to_id)
testdata = TensorDataset(input_ids, attention_masks, label_ids)
testloader = DataLoader(testdata, sampler=RandomSampler(testdata), batch_size=32)

# Optimizer. Adam as recommended by original BERT paper.
optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-5)

loss_values = []  # hold loss values

# Training loop
def train_model(model, optimizer, dataloader, epoch):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch}"):
        batch = tuple(t.to(model.device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

        model.zero_grad()
        optimizer.zero_grad()

        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    loss_values.append(avg_loss)
    print(f"Average Training Loss: {avg_loss}")

# For validation purposes only: Not sure if works as intended.
def compute_accuracy(predictions, labels, mask):
    with torch.no_grad():
        predictions = predictions[mask].flatten()
        labels = labels[mask].flatten()
        correct_predictions = torch.sum(predictions == labels).item()
        total_predictions = mask.sum().item()

    return correct_predictions / total_predictions if total_predictions > 0 else 0

# Validation loop
def validate_model(model, dataloader, epoch):
    model.eval()
    total_loss = 0
    total_accuracy = 0

    for batch in tqdm(dataloader, desc=f"Validation Epoch {epoch}"):
        batch = tuple(t.to(model.device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            labels = inputs["labels"]
            mask = labels != -100  # Ensure we're only calculating accuracy on non-seperator tokens

            accuracy = compute_accuracy(predictions, labels, mask)
            total_accuracy += accuracy

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    print(f"Average Validation Loss: {avg_loss} Accuracy: {avg_accuracy}")

# Train the model, currently 4 epochs.
for epoch in range(4):
    train_model(model, optimizer, dataloader, epoch)
    validate_model(model, testloader, epoch)

# Plot loss as a function of epochs
plt.plot(loss_values, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss as a function of Epochs')
plt.legend()
plt.show()

# Save model for prediction usage
tokenizer.save_pretrained('BERT_model')
model.save_pretrained('BERT_model')
