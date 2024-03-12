import random as r

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

def add_noise_to_text():
  return 1