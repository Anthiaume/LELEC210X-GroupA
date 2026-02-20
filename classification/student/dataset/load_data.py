import os
from os import listdir
from os.path import isfile, join
import pickle

def load_data(locals, speakers, all):
    """
    Docstring for load_data
    
    :param locals: Description
    :param speakers: Description
    :param all: Description
    """
    data = []
    folderpaths = []
    for i in range(len(locals)):
        for speaker in range(len(speakers)):
            folderpaths.append(os.path.join(locals[i], speakers[speaker]))
    for folder in folderpaths:
        mypath = os.path.join(folder)
        files_in_directory = [f for f in listdir(mypath) if (isfile(join(mypath, f)))]
        for file in files_in_directory:
            with open (os.path.join(mypath, file), "rb") as f:
                data.append(pickle.load(f))

    return data