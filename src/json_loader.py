import csv
import json

def load_json_file(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    return data

#csv_file_path = "../ocr_output.json"
#objects = load_json_file(csv_file_path)

#for obj in objects:
#    print(obj)

#print(len(objects))