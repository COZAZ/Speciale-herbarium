import pandas as pd
import numpy as np
import re
import json
from .text_utility import *

# TODO: Finetune (Lave strategier i de forskellige kategorier - f.eks. skal der altid stå Leg, eller kun nogle gange)
# TODO: Find ud af hvilken rækkefølge tingene skal stå i

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
  lats = df[lat].dropna().to_numpy()
  longs = df[lon].dropna().to_numpy()
  filtered_lats = lats[["°" in lat for lat in lats]]
  filtered_longs = longs[["°" in lon for lon in longs]]

  data_columns = [dates, species, dets, locations, legs, filtered_lats, filtered_longs]

  return data_columns

# Create dict and call functions
def createSingleLine(data_list):
  line = {"tokens": [], "labels": []}
  addStartNoise(line)
  selectAndFormatDate(line)
  addRegularnoise(line)
  selectAndFormatSpecies(line, data_list[1])
  selectAndFormatDet(line, data_list[2])
  addRegularnoise(line)
  selectAndFormatLocation(line, data_list[3])
  selectAndFormatLeg(line, data_list[4])
  addRegularnoise(line)
  selectAndFormatCoords(line, data_list[5], data_list[6])
  addEndNoise(line)

  return line

def addStartNoise(dict):
  if is_below_percentage(50):
    dict["tokens"].append(get_random_noise("startnoise"))
    dict["labels"].append("0")

def addRegularnoise(dict):
  # TODO: Maybe add comma to things like "Un 02,201-5"
  if is_below_percentage(33):
    dict["tokens"].append(get_random_noise("regularnoise"))
    dict["labels"].append("0")

def addEndNoise(dict):
  if is_below_percentage(50):
    dict["tokens"].append(get_random_noise("endnoise"))
    dict["labels"].append("0")

# Specifc functions for each of the areas of interest
# TODO: add occasional roman numbers for days
# TODO: add cases for stuff like "Oct. 2015"
# TODO: byt rækkefølge på dag og måned nogle gange
def selectAndFormatDate(dict):
  months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
  
  if is_below_percentage(50):
    random_day = np.random.randint(1,31)
    random_month_name = np.random.choice(months)
    random_year = np.random.randint(1800,2024)

    modified_date = "{0} {1}, {2}".format(random_month_name, random_day, random_year)

  else:
    random_day = np.random.randint(1,32)
    random_month_number = np.random.randint(1,13)
    random_year = np.random.randint(1800,2024)

    if random_day < 10:
      random_day = '0' + str(random_day)
    
    if random_month_number < 10:
      random_month_number = '0' + str(random_month_number)

    modified_date = "{0}-{1}-{2}".format(random_year, random_month_number, random_day)

  dict["tokens"].append(modified_date)
  dict["labels"].append("3")

# TODO: Don't split sepciemen
def selectAndFormatSpecies(dict, species):
  specimen = str(np.random.choice(species))

  # Removes "(current)" from the string
  parts = specimen.split(" (current)")
  specimen_fixed = parts[0].strip()
  tokens = re.split(r'[ ,.]', specimen_fixed)
  """
  # Outcommented for now,
  as it probaly shouldn't be split but maybe some time in the future it should?
  But what do I know, I am just a computer uehuhueuhe
  for token in tokens:
    if token == '':
      continue
    else:
      dict["tokens"].append(token)

    if token == '.' or token == ',':
      dict["labels"].append("0")
    else:
      dict["labels"].append("4")
  """
  dict["tokens"].append(specimen_fixed)
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

def synthesize_text_data(amount, asJson=False):
  data_columns = load_text_data()
  synthesized_text_data = np.zeros(amount)
  synthesized_text_data = list(map(lambda _: createSingleLine(data_columns), synthesized_text_data))

  if asJson == True:
    synthJsonData = json.dumps(synthesized_text_data, indent=2)

    with open("synth_data.json", "w") as outfile:
      outfile.write(synthJsonData)

  return synthesized_text_data

def pretty_print_text_data(token_list):
  for text_obj in token_list:
    print("{0}\n".format(text_obj["tokens"]))