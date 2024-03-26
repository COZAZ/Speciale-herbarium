# Speciale-herbarium
Will you please read me. Please.
This is a test.
For extraction of images: https://specify-attachments.science.ku.dk/static/NHMD_Botany/originals/

# How to run BERT
Follow the steps to run BERT predictions on your local machine.

Step 1:
Go into gen_data.py and modify the amount according to the number of training samples wanted.
$ python gen_data.py 

Step 2:
Train the model
$ python solo_bert.py

Step 3:
Run predictions
$ python pred.py