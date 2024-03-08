import pandas as pd
import numpy as np
import re

# TODO:
# Fix splitting of tokens and assign correct amount of labels.
# Fix coordinate handling
# Fix filler tokens
# Finetune

# load initial dataset
df = pd.read_csv("greenland.csv")

date = "1,10.collectingevent.startDate"
spec = "1,9-determinations.collectionobject.determinations"
det = "1,9-determinations,5-determiner.determination.determiner"
loc = "1,10,2.locality.localityName"
leg = "1,10,30-collectors.collectingevent.collectors"
lat = "1,10,2.locality.lat1text"
lon = "1,10,2.locality.long1text"

# Extract columns we are interested in
dates = df[date].dropna().to_numpy()
species = df[spec].dropna().to_numpy()
dets = df[det].dropna().to_numpy()
locations = df[loc].dropna().to_numpy()
legs = df[leg].dropna().to_numpy()
# TODO: Fix coordinates - Done??
lats = df[lat].dropna().to_numpy()
longs = df[lon].dropna().to_numpy()
filtered_lats = lats[["°" in lat for lat in lats]]
filtered_longs = longs[["°" in lon for lon in longs]]

# Create dict and call functions
def createSingleLine():
  line = {"tokens": [], "labels": []}
  selectAndFormatDate(line)
  selectAndFormatSpecies(line)
  selectAndFormatDet(line)
  selectAndFormatLocation(line)
  selectAndFormatLeg(line)
  selectAndFormatCoords(line)

  return line

# Specifc functions for each of the areas of interest
def selectAndFormatDate(dict):
  date = str(np.random.choice(dates))
  dict["tokens"].append(date)
  dict["labels"].append("3")

def selectAndFormatSpecies(dict):
  specimen = str(np.random.choice(species))
  # Removes "(current)" from the string
  # TODO: Does this lead to an empty string in every species case and should it be removed then?
  parts = specimen.split("(current)")
  specimen_fixed = parts[0]
  tokens = re.split(r'[ ,.]', specimen_fixed)
  
  for token in tokens[:-1]: # Remove empty string from name???
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

def selectAndFormatCoords(dict):
  lat = str(np.random.choice(filtered_lats))
  lon = str(np.random.choice(filtered_longs))
  dict["tokens"].append(lat + ', ' + lon)
  dict["labels"].append("6")

def synthesize_text_data():
  synthesized_text_data = np.zeros(100)
  synthesized_text_data = list(map(lambda _: createSingleLine(), synthesized_text_data))

  return synthesized_text_data