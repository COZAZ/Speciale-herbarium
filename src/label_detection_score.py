def compute_bounding_box_score(label_data):

    for label_object in label_data:
        im_name = label_object[1]
        labe_loc = (label_object[0])[1:]

        print(im_name)
        print(labe_loc)

        



    return label_data