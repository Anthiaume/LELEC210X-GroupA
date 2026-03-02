import os
from os import listdir
from os.path import isfile, join
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def load_data(records):
    """
    Inputs:
    - records : list of tuples (mcu, local, speaker) specifying the data to load. For example:
        Example : records = [("mcu13", "vinikot", "JBL Flip 5 - Auguste - spec_20_20"),
                             ("mcu13", "fisher", "local speakers - spec_20_20")       ]
    Output:
    - data: numpy array of melspectrograms
    - labels: numpy array of labels (e.g. "chainsaw", "crackling fire", "fireworks", "gunshot", "background")
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

def save_confusion_matrix(model, x_test, y_test, filename="CONFUSION_MATRIX.pdf", show=True):
    # 1. Génération de la matrice
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        x_test,
        y_test,
        display_labels=model.classes_,
        cmap=plt.cm.Blues,
        normalize=None,
    )

    # 2. Ajustement des axes (Utilisation de .ax_)
    ticks = range(len(model.classes_))

    # Correction : disp.ax_ au lieu de disp.ax
    disp.ax_.set_xticks(ticks)
    disp.ax_.set_xticklabels(model.classes_, rotation=45, ha='right', rotation_mode='anchor')

    disp.ax_.set_yticks(ticks)
    disp.ax_.set_yticklabels(model.classes_)

    # 3. Augmenter la taille des labels des axes
    disp.ax_.set_xlabel('Predicted labels', fontsize=14, fontweight='bold')
    disp.ax_.set_ylabel('True labels', fontsize=14, fontweight='bold')

    # 4. Ajustement automatique pour ne pas couper les labels inclinés
    plt.tight_layout()

    # 5. Sauvegarde et affichage
    plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()