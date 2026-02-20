import argparse
import os 

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

LOCAL = "sud11"
SPEAKER = "local_speakers"
TYPE = "chainsaw" # "chainsaw", "gunshot", "fireworks", "crackling fire", "background"
save_name = None # None, "xxx.pdf"
plot_type = "single" # "single" or "multiple"
multiple_plot_dimension = (3, 3)


if __name__ == "__main__":
    mypath = os.path.join("dataset", LOCAL, SPEAKER, f"{TYPE}.pkl",)

    with open(mypath, "rb") as f:
        melvec = pickle.load(f)

    if plot_type == "single":
        plt.figure()
        plot_specgram(
            melvec.reshape((N_MELVECS, MELVEC_LENGTH)).T,
            ax=plt.gca(),
            is_mel=True,
            title=f"MEL Spectrogram : {save_name}",
            xlabel="Mel vector",
        )
        #plt.show()
        if save_name:
            plt.savefig(save_name)
        plt.close()

    elif plot_type == "multiple":
        fig, axes = plt.subplots(*multiple_plot_dimension, figsize=(12, 12))
        for i in range(multiple_plot_dimension[0]):
            for j in range(multiple_plot_dimension[1]):
                idx = i * multiple_plot_dimension[1] + j
                if idx < melvec.shape[0]:
                    plot_specgram(
                        melvec[idx].reshape((N_MELVECS, MELVEC_LENGTH)).T,
                        ax=axes[i, j],
                        is_mel=True,
                        title=f"Mel spectrogram {idx}",
                        xlabel="Mel vector",
                    )
                else:
                    axes[i, j].axis("off")
        plt.tight_layout()
        #plt.show()
        if save_name:
            plt.savefig(save_name)
        plt.close()