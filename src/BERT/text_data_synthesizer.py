import pandas as pd
import numpy as np
import re
from .text_utility import *

# TODO:
# Fix splitting of tokens and assign correct amount of labels.
# Fix coordinate handling
# Fix filler tokens
# Finetune

def load_text_data():
  # load initial dataset
  df = pd.read_csv("../greenland.csv")

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

  data_columns = [dates, species, dets, locations, legs, filtered_lats, filtered_longs]

  return data_columns

# Create dict and call functions
def createSingleLine(data_list):
  line = {"tokens": [], "labels": []}
  selectAndFormatDate(line, data_list[0])
  selectAndFormatSpecies(line, data_list[1])
  selectAndFormatDet(line, data_list[2])
  selectAndFormatLocation(line, data_list[3])
  selectAndFormatLeg(line, data_list[4])
  selectAndFormatCoords(line, data_list[5], data_list[6])

  return line

# Specifc functions for each of the areas of interest
def selectAndFormatDate(dict, dates):
  date = str(np.random.choice(dates))
  dict["tokens"].append(date)
  dict["labels"].append("3")

def selectAndFormatSpecies(dict, species):
  specimen = str(np.random.choice(species))
  # Removes "(current)" from the string
  # TODO: Does this lead to an empty string in every species case and should it be removed then?
  parts = specimen.split("(current)")
  specimen_fixed = parts[0]
  tokens = re.split(r'[ ,.]', specimen_fixed)
  
  for token in tokens: # Remove empty string from name - done??
    if token == '':
      continue
    else:
      dict["tokens"].append(token)

    if token == '.' or token == ',':
      dict["labels"].append("0")
    else:
      dict["labels"].append("4")

def selectAndFormatDet(dict, dets):
  det = str(np.random.choice(dets))

  if is_below_percentage(20) and (',' in det):
    name = det
    names = name.split(',')
    det = names[1].strip() + " " + names[0].strip()
  
  if is_below_percentage(70):
    det = "Det: " + det

  dict["tokens"].append(det)
  dict["labels"].append("5")

def selectAndFormatLeg(dict, legs):
  leg = str(np.random.choice(legs))

  if is_below_percentage(20) and (',' in leg):
    name = leg
    names = name.split(',')
    leg = names[1].strip() + " " + names[0].strip()

  if is_below_percentage(80):
    leg = "Leg: " + leg

  dict["tokens"].append(leg)
  dict["labels"].append("1")

def selectAndFormatLocation(dict, locations):
  location = str(np.random.choice(locations))
  location = location.replace(u'\xa0', u' ')

  location_parts = location.split(' ')

  if is_below_percentage(50):
    location_parts = location.split(' ')
    location = shuffle_content(location_parts)

  dict["tokens"].append(location)
  dict["labels"].append("2")

def selectAndFormatCoords(dict, filtered_lats, filtered_longs):
  lat = str(np.random.choice(filtered_lats))
  lon = str(np.random.choice(filtered_longs))
  dict["tokens"].append(lat + ', ' + lon)
  dict["labels"].append("6")

def synthesize_text_data(amount):
  data_columns = load_text_data()

  synthesized_text_data = np.zeros(amount)
  synthesized_text_data = list(map(lambda _: createSingleLine(data_columns), synthesized_text_data))

  return synthesized_text_data

def pretty_print_text_data(token_list):
  for text_obj in token_list:
    print("{0}\n".format(text_obj["tokens"]))