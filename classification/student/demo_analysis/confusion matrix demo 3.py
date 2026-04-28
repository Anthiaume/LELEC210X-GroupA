import pickle
from xml.parsers.expat import model

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def save_confusion_matrix(confusion_matrix=None, filename="CONFUSION_MATRIX.pdf", show=True, labels=None):
    # 1. Génération de la matrice
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=labels,
        )

    disp.plot(cmap=plt.cm.Blues)

    # 2. Ajustement des axes (Utilisation de .ax_)
    ticks = range(len(labels))

    # Correction : disp.ax_ au lieu de disp.ax
    disp.ax_.set_xticks(ticks)
    disp.ax_.set_xticklabels(labels, rotation=45, ha='right', rotation_mode='anchor')

    disp.ax_.set_yticks(ticks)
    disp.ax_.set_yticklabels(labels)

    # 3. Augmenter la taille des labels des axes
    disp.ax_.set_xlabel('Predicted labels', fontsize=14, fontweight='bold')
    disp.ax_.set_ylabel('True labels', fontsize=14, fontweight='bold')

    # 4. Ajustement automatique pour ne pas couper les labels inclinés
    plt.tight_layout()

    # 5. Sauvegarde et affichage
    plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()

with open("interesting_data.pkl", "rb") as f:
    interesting_data = pickle.load(f)

zeros = np.zeros((5,5))
labels = ["background", "chainsaw", "fire", "fireworks", "gunshot"]
for i in range(len(interesting_data)):
    if interesting_data[i][1]  in labels:
        if interesting_data[i][2] in labels:
            print(interesting_data[i][1], interesting_data[i][2])
            zeros[labels.index(interesting_data[i][2]), labels.index(interesting_data[i][1])] += 1

save_confusion_matrix(zeros, filename="CONFUSION_MATRIX_DEMO_3.pdf", show=False, labels=labels)

# Compute accuracy
accuracy = np.trace(zeros) / np.sum(zeros)
print(f"Accuracy: {accuracy:.4f}")