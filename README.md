# Master Thesis - Digitization of Herbarium Samples

# How to run pipeline on images:
Step 1:
To execute the pipeline, you must have a directory of .jpg files ready.

Step 2:
For this prototype, you must then place this folder inside our **Speciale-herbarium** repo.

Step 3:
Within **src**, inside our **run.py** file, set the **image_directory** variable to the name of your folder with images.

Step 4:
In your terminal, navigate to to the **src** directory of our codebase and run the command **python3 run.py** to start the pipeline process.

NOTE:
By default, a folder named **herb_images_1980** is already placed within **Speciale-herbarium**, as the used images when running the pipeline.
To run the pipeline on other images, please follows the above steps 1-4.

#  How to fine-tune the BERT model if needed.
If the need to train a new NER model arises, please follow the steps below.

## Step 1:
Go into gen_data.py and modify the amount according to the number of training samples wanted.
Set the 'asJson' parameter to 'True'.
$ python gen_data.py 

## Step 2:
Train the model. The .json file for training data should be in the right folder from step 1.
Uses the pre-trained model: https://huggingface.co/google-bert/bert-base-multilingual-cased
$ python solo_bert.py

## Step 3 (Optional):
The model has now been trained, it will test YOLO, OCR and BERT steps. It requires them all to be completed, to produce test results.
Specifically, the testing of BERT will include validation on synthetic data, but also predictions on OCR produced data from the output CSV files.
$ python testing.py