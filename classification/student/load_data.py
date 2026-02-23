import os
from os import listdir
from os.path import isfile, join
import pickle
from tkinter.font import names
import pandas as pd

def load_data(locals, speakers):
    """
    Inputs:
    - locals: list of strings, each string is the name of a local (e.g. "vinikot")
    - speakers: list of strings, each string is the name of a speaker (e.g. "JBL Flip 5 - Auguste")
    Output:
    - df: pandas dataframe with two columns: "melspecgram" and "label". The "melspecgram" column contains the melspectrograms of the audio files, and the "label" column contains the labels of the audio files (e.g. "vinikot", "JBL Flip 5 - Auguste")
    """
    data = []
    labels = []
    folderpaths = []

    # parcours tous les locaux et tous les speakers
    for i in range(len(locals)):
        for speaker in range(len(speakers)):
            if os.path.exists(os.path.join(os.getcwd(), "dataset", locals[i], speakers[speaker])):
                folderpaths.append(os.path.join(os.getcwd(), "dataset", locals[i], speakers[speaker]))

    for folder in folderpaths:
        mypath = folder
        files_in_directory = [f for f in listdir(mypath) if (isfile(join(mypath, f)))]
        for file in files_in_directory:
            with open (os.path.join(mypath, file), "rb") as f:
                data.append(pickle.load(f))
                labels.append(file.split("_")[0])
    #create a pandas dataframe with the data and the labels
    df = pd.DataFrame({'melspecgram': data, 'label': labels})

    return df