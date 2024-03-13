import random as r
import pandas as pd
import numpy as np

def is_below_percentage(percent):
  random_val = r.randint(1,100)
  if random_val <= percent:
    return True
  else:
    return False

def shuffle_content(list):
  r.shuffle(list)
  concatenated_string = " ".join(list)

  return concatenated_string

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