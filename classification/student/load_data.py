import os
from os import listdir
from os.path import isfile, join
import pickle
import numpy as np

def load_data(records):
    """
    Inputs:
    - records : list of tuples (mcu, local, speaker) specifying the data to load. For example:
        Example : records = [("mcu13", "vinikot", "JBL Flip 5 - Auguste - spec_20_20"),
                             ("mcu13", "fisher", "local speakers - spec_20_20")       ]
    Output:
    - data: numpy array of melspectrograms
    - labels: numpy array of labels (e.g. "vinikot", "JBL Flip 5 - Auguste")
    """
    data = []
    labels = []
    folderpaths = []

    # parcours tous les locaux et tous les speakers
    for i in range(len(records)):
        if os.path.exists(os.path.join(os.getcwd(), "dataset", records[i][0], records[i][1], records[i][2])):
            folderpaths.append(os.path.join(os.getcwd(), "dataset", records[i][0], records[i][1], records[i][2]))

    for folder in folderpaths:
        mypath = folder
        files_in_directory = [f for f in listdir(mypath) if (isfile(join(mypath, f)))]
        for file in files_in_directory:
            with open (os.path.join(mypath, file), "rb") as f:
                data.append(pickle.load(f))
                labels.append(file.split("_")[0])
                
    data = np.array(data)
    labels = np.array(labels)

    return data, labels