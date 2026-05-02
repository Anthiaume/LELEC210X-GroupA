import argparse
from genericpath import isfile
import os
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import serial
from serial.tools import list_ports
import pickle, time

from classification.utils.plots import plot_specgram

PRINT_PREFIX = "DF:HEX:"  
FREQ_SAMPLING = 10200
MELVEC_LENGTH = 20
N_MELVECS = 20

types = ["chainsaw", "gunshot", "fireworks", "crackling fire", "background"]

for TYPE in types:
#TYPE = "crackling fire" # "chainsaw", "gunshot", "fireworks", "crackling fire", "background"
    NUMBER = 1 # number of the melspectrogram to plot, only for single plot
    plot_type = "all" # "single" or "all"
    multiple_plot_dimension = 4 # It is a squared plot, so it is the number of rows and columns (e.g., 4 means 4x4 = 16 plots)



    mypath = os.getcwd()#os.path.join("dataset", LOCAL, SPEAKER)
    onlyfiles = [f  for f in os.listdir(mypath) if (isfile(join(mypath, f)) and f.endswith(".pkl") and f.startswith(TYPE))]
    # shuffle onlyfiles to get random melspectrograms each time
    np.random.shuffle(onlyfiles)
    
    data = []
    for i in range(len(onlyfiles)):
        with open(os.path.join(mypath, onlyfiles[i]), "rb") as f:
            melvec = pickle.load(f)
            data.append(melvec)

    plots = 0
    while plots < len(data):
        fig, axes = plt.subplots(multiple_plot_dimension, multiple_plot_dimension, figsize=(12, 12))
        for i in range(multiple_plot_dimension):
            for j in range(multiple_plot_dimension):
                idx = i * multiple_plot_dimension + j + plots
                if idx < len(data):
                    plot_specgram(
                        data[idx].reshape((N_MELVECS, MELVEC_LENGTH)).T,
                        ax=axes[i, j],
                        is_mel=True,
                        title=f"{TYPE} - {idx+1}",
                        xlabel="Mel vector",
                    )
                else:
                    axes[i, j].axis("off")
        # save if the square is full or if it's the last row
        plt.tight_layout()
        plt.savefig(f"Melspectrograms beauty - {TYPE} - part {plots // 16}.pdf")
        plots += 16
        plt.close()

