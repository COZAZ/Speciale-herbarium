import pandas as pd
import numpy as np
import re

# TODO:
# Fix splitting of tokens and assign correct amount of labels.
# Fix coordinate handling
# Fix filler tokens
# Finetune

# load initial dataset
df = pd.read_excel("greenland.xlsx")

# Extract columns we are interested in
dates = df["date"].dropna().to_numpy()
species = df["art"].dropna().to_numpy()
dets = df["det"].dropna().to_numpy()
locations = df["location"].dropna().to_numpy()
legs = df["collector"].dropna().to_numpy()
# TODO: Fix coordinates

# Create dict and call functions
def createSingleLine():
  line = {"tokens": [], "labels": []}
  selectAndFormatDate(line)
  selectAndFormatSpecies(line)
  selectAndFormatDet(line)
  selectAndFormatLocation(line)
  selectAndFormatLeg(line)
  print(line)


# Specifc functions for each of the areas of interest
def selectAndFormatDate(dict):
  date = str(np.random.choice(dates))
  dict["tokens"].append(date)
  dict["labels"].append("3")

def selectAndFormatSpecies(dict):
  specimen = str(np.random.choice(species))
  parts = specimen.split("(current)")
  specimen_fixed = parts[0]
  tokens = re.split(r'[ ,.]', specimen_fixed)
  for token in tokens:
    dict["tokens"].append(token)
    if token == '.' or token == ',':
      dict["labels"].append("0")
    else:
      dict["labels"].append("4")

def selectAndFormatDet(dict):
  det = str(np.random.choice(dets))
  dict["tokens"].append(det)
  dict["labels"].append("5")

def selectAndFormatLocation(dict):
  location = str(np.random.choice(locations))
  dict["tokens"].append(location)
  dict["labels"].append("2")

def selectAndFormatLeg(dict):
  leg = str(np.random.choice(legs))
  dict["tokens"].append(leg)
  dict["labels"].append("1")


createSingleLine()
