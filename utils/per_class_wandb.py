
KAIST_CLASSES =  ['person', 'people', 'cyclist', 'person?']

def ap_per_class_to_wandb_format(nt,dataset_used, p, r, ap, f1):
    """ 
    nt: should be per class instance of object, [1019, 0, 190,0]
    dataset: dataset type
    p: precision
    r: recall
    ap: average precison of each class is a different row  
    """

    ap50, ap75, ap = ap[:, 0], ap[:, 5], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    mp, mr, map50, map75, map = p.mean(), r.mean(), ap50.mean(), ap75.mean(), ap.mean()
    i = 0
    row_i = 0 # updated like this in case we do not have a class 
    map50_person = map75_person = map_person = map50_people = map75_people = map_people = \
    map50_cyclist = map75_cyclist = map_cyclist = map50_person_question = map75_person_question = map_person_question = -1
    if dataset_used == 'kaist':
        for i in range(nt.size):
            if i == 0:
                if nt[i] == 0:
                    map50_person, map75_person, map_person = -1, -1, -1
                else:
                    map50_person, map75_person, map_person = ap50[row_i], ap75[row_i], ap[row_i]
            elif i == 1:
                if nt[i] == 0:
                    map50_people, map75_people, map_people = -1, -1, -1
                else:
                    map50_people, map75_people, map_people = ap50[row_i], ap75[row_i], ap[row_i]
            elif i == 2:
                if nt[i] == 0:
                    map50_cyclist, map75_cyclist, map_cyclist = -1, -1, -1
                else:
                    map50_cyclist, map75_cyclist, map_cyclist = ap50[row_i], ap75[row_i], ap[row_i]
            elif i == 3:
                if nt[i] == 0:
                    map50_person_question, map75_person_question, map_person_question = -1, -1, -1
                else:
                    map50_person_question, map75_person_question, map_person_question = ap50[row_i], ap75[row_i], ap[row_i]

            if nt[i] != 0:
                row_i += 1
        # map50_person, map50_people, map50_cyclist, map50_person_question = ap50[0],ap50[1], ap50[2], ap50[3]
        # map75_person, map75_people, map75_cyclist, map75_person_question = ap75[0],ap75[1], ap75[2], ap75[3]
        # map_person, map_people, map_cyclist, map_person_question = ap[0], ap[1], ap[2], ap[3] #AP@0.5:0.95

        return map50_person, map50_people, map50_cyclist, map50_person_question, \
                map75_person, map75_people, map75_cyclist, map75_person_question, \
                map_person, map_people, map_cyclist, map_person_question, \
                ap50, ap75, ap, mp, mr, map50, map75, map 
    else:
        return ap50, ap75, ap, mp, mr, map50, map75, map








    
    