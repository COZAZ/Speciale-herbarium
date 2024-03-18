import pandas as pd
import numpy as np
import json
from .text_utility import *

def load_text_data():
  # load initial dataset
  df = pd.read_csv("../greenland.csv", dtype=str)

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
  filtered_lats = lats[['°' in lat for lat in lats]]
  filtered_longs = longs[['°' in lon for lon in longs]]

  data_columns = [dates, species, dets, locations, legs, filtered_lats, filtered_longs]

  return data_columns

# Create dict and call functions
def createSingleLine(data_list):
  line = {"tokens": [], "labels": []}

  # Order of tokens based on general assesment of image layout
  addStartNoise(line)
  addRegularnoise(line)
  selectAndFormatSpecies(line, data_list[1])
  addRegularnoise(line)
  selectAndFormatLocation(line, data_list[3])
  selectAndFormatCoords(line, data_list[5], data_list[6])
  addRegularnoise(line)
  selectAndFormatDate(line)
  selectAndFormatLeg(line, data_list[4])
  selectAndFormatDet(line, data_list[2])
  addEndNoise(line)

  return line

def addStartNoise(dict):
  if is_below_percentage(50):
    dict["tokens"].append(get_random_noise("startnoise"))
    dict["labels"].append("0")

def addRegularnoise(dict):
  if is_below_percentage(50):
    dict["tokens"].append(get_random_noise("regularnoise"))
    dict["labels"].append("0")

def addEndNoise(dict):
  if is_below_percentage(50):
    dict["tokens"].append(get_random_noise("endnoise"))
    dict["labels"].append("0")

# Specifc functions for each of the areas of interest
def selectAndFormatDate(dict):
  months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
  abbreviated_months = ["Jan.", "Feb.", "Mar.", "Apr.", "May.", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec."]
  roman_number_days = ["III", "IV", "VI", "VII", "IX", "XVI", "XXIV", "XVII", "XXXI", "XXX"]
  
  if is_below_percentage(50):
    if is_below_percentage(70):
      random_day = np.random.randint(1,31)
    else:
      random_day = np.random.choice(roman_number_days)
    random_month_name = np.random.choice(months)
    random_year = np.random.randint(1800,2024)

    if is_below_percentage(50):
      modified_date = "{0} {1}, {2}".format(random_day, random_month_name, random_year)
    else:
      modified_date = "{0} {1}, {2}".format(random_month_name, random_day, random_year)

  else:
    random_day = np.random.randint(1,32)
    random_month_number = np.random.randint(1,13)
    random_year = np.random.randint(1800,2024)

    if random_day < 10:
      random_day = '0' + str(random_day)
    
    if random_month_number < 10:
      random_month_number = '0' + str(random_month_number)

    if is_below_percentage(50):
      modified_date = "{0}-{1}-{2}".format(random_year, random_month_number, random_day)
    else:
      modified_date = "{0}-{1}-{2}".format(random_year, random_day, random_month_number)

  if is_below_percentage(20):
    modified_date = "{0} {1}".format(np.random.choice(abbreviated_months), random_year)

  dict["tokens"].append(modified_date)
  dict["labels"].append("B-DATE")

def selectAndFormatSpecies(dict, species):
  specimen = str(np.random.choice(species))

  # Removes "(current)" from the string
  parts = specimen.split(" (current)")
  specimen_fixed = parts[0].strip()
  """
  tokens = re.split(r'[ ,.]', specimen_fixed)
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
  dict["labels"].append("B-SPECIMEN")

def selectAndFormatDet(dict, dets):
  det = str(np.random.choice(dets))

  if is_below_percentage(20) and (',' in det):
    name = det
    names = name.split(',')
    det = names[1].strip() + " " + names[0].strip()

    """
    if is_below_percentage(10) and np.any(names[0] != ' ') and np.any(names[1] != ' '):
      det = name_to_initials(names)
    """
      
  if is_below_percentage(50):
    if is_below_percentage(50):
      det = "Det: " + det
    else:
      det = "determ: " + det

  dict["tokens"].append(det)
  dict["labels"].append("B-DET")

def selectAndFormatLeg(dict, legs):
  leg = str(np.random.choice(legs))

  if is_below_percentage(20) and (',' in leg):
    name = leg
    names = name.split(',')
    leg = names[1].strip() + " " + names[0].strip()

    """
    if is_below_percentage(10) and np.any(names[0] != ' ') and np.any(names[1] != ' '):
      leg = name_to_initials(names)
    """

  if is_below_percentage(50):
    if is_below_percentage(50):
      leg = "Leg: " + leg
    else:
      leg = "legit: " + leg

  dict["tokens"].append(leg)
  dict["labels"].append("B-LEG")

def selectAndFormatLocation(dict, locations):
  location = str(np.random.choice(locations))
  location = location.replace(u'\xa0', u' ')

  if is_below_percentage(35):
    location_parts = location.split(' ')
    location = shuffle_content(location_parts)

  dict["tokens"].append(location)
  dict["labels"].append("B-LOCATION")

def selectAndFormatCoords(dict, filtered_lats, filtered_longs):
  lat = str(np.random.choice(filtered_lats))
  lon = str(np.random.choice(filtered_longs))
  
  if is_below_percentage(50):
    lat = get_random_lat()

  if is_below_percentage(50):
    lon = get_random_lon()
  
  if is_below_percentage(15):
    coord_set = lon + ', ' + lat
  else:
    coord_set = lat + ', ' + lon

  dict["tokens"].append(coord_set)
  dict["labels"].append("B-COORD")

def synthesize_text_data(amount, asJson=False):
  data_columns = load_text_data()
  synthesized_text_data = np.zeros(amount)
  synthesized_text_data = list(map(lambda _: createSingleLine(data_columns), synthesized_text_data))

  if asJson:
    synthJsonData = json.dumps(synthesized_text_data, indent=2)

    with open("synth_data.json", "w") as outfile:
      outfile.seek(0)
      outfile.truncate()
      outfile.write(synthJsonData)

  return synthesized_text_data

def pretty_print_text_data(token_list):
  print("\nSynthetic text for BERT:")
  for text_obj in token_list:
    print("TEXT: {0}\nLABELS: {1}\n".format(text_obj["tokens"], text_obj["labels"]))