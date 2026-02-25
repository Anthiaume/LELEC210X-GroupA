import os
from os import listdir
from os.path import isfile, join
import pickle
import numpy as np

def load_data(mcu, locals, speakers):
    """
    Inputs:
    - mcu: string, the name of the MCU (e.g. "mcu13")
    - locals: list of strings, each string is the name of a local (e.g. "vinikot")
    - speakers: list of strings, each string is the name of a speaker (e.g. "JBL Flip 5 - Auguste")
    Output:
    - data: numpy array of melspectrograms
    - labels: numpy array of labels (e.g. "vinikot", "JBL Flip 5 - Auguste")
    """
    data = []
    labels = []
    folderpaths = []

    # parcours tous les locaux et tous les speakers
    for i in range(len(locals)):
        for speaker in range(len(speakers)):
            if os.path.exists(os.path.join(os.getcwd(), "dataset", mcu, locals[i], speakers[speaker])):
                folderpaths.append(os.path.join(os.getcwd(), "dataset", mcu, locals[i], speakers[speaker]))

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