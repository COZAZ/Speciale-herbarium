import random as r
import pandas as pd
import numpy as np

def is_below_percentage(percent):
  random_val = r.randint(1,100)
  if random_val <= percent:
    return True
  else:
    return False

def get_random_noise(noiseType):
  # load initial dataset
  df = pd.read_csv("../noise.csv")

  # Extract columns we are interested in
  startnoise = df["startnoise"].dropna().to_numpy()
  endnoise = df["endnoise"].dropna().to_numpy()
  regularnoise = df["regularnoise"].dropna().to_numpy()

  if noiseType == "startnoise":
    return str(np.random.choice(startnoise))
  elif noiseType == "endnoise":  
    return str(np.random.choice(endnoise))
  elif noiseType == "regularnoise":
    return str(np.random.choice(regularnoise))

def get_random_lat():
  degree = r.randint(1,90)
  direction = r.choice(['S','N'])
  minute = round(r.uniform(1,60), 2)

  random_lat = str(degree) + '° ' + str(minute) + '\' ' + direction

  return random_lat

def get_random_lon():
  degree = r.randint(1,180)
  direction = r.choice(['E','W'])
  minute = round(r.uniform(1,60), 2)

  random_lon = str(degree) + '° ' + str(minute) + '\' ' + direction

  return random_lon