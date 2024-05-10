import pandas as pd
from thefuzz import fuzz

def compare_csv(input_csv, true_csv, metric): 
    specimen_similarity = 0
    location_similarity = 0
    legit_similarity = 0
    determinant_similarity = 0
    date_similarity = 0
    coordinates_similarity = 0

    spec_skips = 0
    loc_skips = 0
    leg_skips = 0
    det_skips = 0
    date_skips = 0
    coord_skips = 0

    for i in range(len(input_csv)):
        current_post_entry = input_csv.loc[i]
        post_id = current_post_entry["Catalog number"]
        
        current_true_entry = true_csv[true_csv["Catalog number"] == post_id]

        # Specimen
        post_specimen = current_post_entry["Specimen"]
        true_specimen = current_true_entry["Specimen"].iloc[0]
        if pd.isna(post_specimen) == True and pd.isna(true_specimen) == True:
            spec_skips += 1
        else:
            if metric == "fuzz":
                spec_ratio = fuzz.ratio(post_specimen, true_specimen)
            elif metric == "char":
                spec_ratio = character_error_rate(post_specimen, true_specimen)
            else: print("Error: no metric")

            specimen_similarity += spec_ratio

        # Location
        post_location = current_post_entry["Location"]
        true_location = current_true_entry["Location"].iloc[0]
        if pd.isna(post_location) == True and pd.isna(true_location) == True:
            loc_skips += 1
        else:
            if metric == "fuzz":
                loc_ratio = fuzz.ratio(post_location, true_location)
            elif metric == "char":
                loc_ratio = character_error_rate(post_location, true_location)
            else: print("Error: no metric")
            
            location_similarity += loc_ratio

        # Legit
        post_legit = current_post_entry["Legit"]
        true_legit = current_true_entry["Legit"].iloc[0]
        if pd.isna(post_legit) == True and pd.isna(true_legit) == True:
            leg_skips += 1
        else:
            if metric == "fuzz":
                leg_ratio = fuzz.ratio(post_legit, true_legit)
            elif metric == "char":
                leg_ratio = character_error_rate(post_legit, true_legit)
            else: print("Error: no metric")
            
            legit_similarity += leg_ratio
        
        # Determinant
        post_determinant = current_post_entry["Determinant"]
        true_determinant = current_true_entry["Determinant"].iloc[0]
        if pd.isna(post_determinant) == True and pd.isna(true_determinant) == True:
            det_skips += 1
        else:
            if metric == "fuzz":
                det_ratio = fuzz.ratio(post_determinant, true_determinant)
            elif metric == "char":
                det_ratio = character_error_rate(post_determinant, true_determinant)
            else: print("Error: no metric")
            
            determinant_similarity += det_ratio
        
        # Date
        post_date = current_post_entry["Date"]
        true_date = current_true_entry["Date"].iloc[0]
        if pd.isna(post_date) == True and pd.isna(true_date) == True:
            date_skips += 1
        else:
            if metric == "fuzz":
                date_ratio = fuzz.ratio(post_date, true_date)
            elif metric == "char":
                date_ratio = character_error_rate(post_date, true_date)
            else: print("Error: no metric")
            
            date_similarity += date_ratio
        
        # Coordinates
        post_coordinates = current_post_entry["Coordinates"]
        true_coordinates = current_true_entry["Coordinates"].iloc[0]
        if pd.isna(post_coordinates) == True and pd.isna(true_coordinates) == True:
            coord_skips += 1
        else:
            if metric == "fuzz":
                coord_ratio = fuzz.ratio(post_coordinates, true_coordinates)
            elif metric == "char":
                coord_ratio = character_error_rate(post_coordinates, true_coordinates)
            else: print("Error: no metric")
            
            coordinates_similarity += coord_ratio

    avg_specimen_similarity = specimen_similarity / (len(input_csv)-spec_skips)
    avg_location_similarity = location_similarity / (len(input_csv)-loc_skips)
    avg_legit_similarity = legit_similarity / (len(input_csv)-leg_skips)
    avg_determinant_similarity = determinant_similarity / (len(input_csv)-det_skips)
    avg_date_similarity = date_similarity / (len(input_csv)-date_skips)
    avg_coordinates_similarity = coordinates_similarity / (len(input_csv)-coord_skips)

    avg_total = (avg_specimen_similarity+avg_location_similarity+avg_legit_similarity+avg_determinant_similarity+avg_date_similarity+avg_coordinates_similarity)/6

    return avg_specimen_similarity, avg_location_similarity, avg_legit_similarity, avg_determinant_similarity, avg_date_similarity, avg_coordinates_similarity, avg_total

def character_error_rate(string1, string2):
    longest_string = ""
    shortest_string = ""
    if len(str(string1)) > len(str(string2)):
        longest_string = string1
        shortest_string = string2
    else:
        longest_string = string2
        shortest_string = string1

    matches = 0
    for i, char in enumerate(str(shortest_string)):
        if char == longest_string[i]:
            matches += 1

    return (matches / len(longest_string)) * 100