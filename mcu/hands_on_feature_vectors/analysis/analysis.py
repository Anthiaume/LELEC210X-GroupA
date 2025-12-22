import numpy as np
import matplotlib.pyplot as plt
import pickle
from classification.utils.plots import plot_specgram


data_folder = "./data acquired/"
classes = ["helicopter", "handsaw", "birds", "chainsaw", "fire"]
MELVEC_LENGTH = 20
N_MELVECS = 20

for i in range(5):
    liste = []
    for j in range(40):
        with open(f"{data_folder}{classes[i]}_{j+1}.pkl", "rb") as f:
                melvec = pickle.load(f)
        normalized_melvec = melvec.reshape(1, -1) / np.linalg.norm(melvec.reshape(1, -1))
        liste.append(normalized_melvec.reshape(normalized_melvec.shape[1],))
	
    # make a mean of all melvecs in liste
    liste = np.array(liste)
    print(liste.shape)

    # J'ai vérifié, ça calcule bien la bonne moyenne
    mean_melvec_np = np.mean(liste, axis=0)

    plt.figure()
    plot_specgram(
        mean_melvec_np.reshape((N_MELVECS, MELVEC_LENGTH)).T,
        ax=plt.gca(),
        is_mel=True,
        title=f"MEL Spectrogram np#{classes[i]}",
        xlabel="Mel vector",
    )
    plt.show()
    plt.savefig(f"Analyse_{classes[i]}.pdf")
    plt.show()
    plt.close()

