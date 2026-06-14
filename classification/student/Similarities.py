from sklearn.model_selection import train_test_split
from torchgen import model

from student_fct import *
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
#import MLP classifier
from sklearn.neural_network import MLPClassifier

records = [("mcu13", "fisher", "local speakers - spec_20_20"),         # 5 classes, 122 samples per class
            ("mcu13", "vinikot", "JBL Flip 5 - Auguste - spec_20_20"),
            ("mcu13", "sud5", "local speakers - spec_20_20"),       # 5 classes, 122 samples per class
            ("mcu13", "sud11", "local speakers - spec_20_20 - Jonathan"),
            ("mcu13", "sud11", "local speakers - spec_20_20 - Raphael")]         # 5 classes, 122 samples per class]

# records = [("mcu13", "sud11", "local speakers - spec_20_20 - Raphael")]         # 5 classes, 122 samples per class]

# Load data and labels
data, labels, labels_encoded, MLP_halikarnassos_classes = load_data(records, classes=["background", "chainsaw", "crackling fire", "fireworks", "gunshot"])

data = data / np.linalg.norm(data, axis=1, keepdims=True)


from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np

# ─── Paramètres ────────────────────────────────────────────────────────────────
N_PAIRS   = 2       # nombre de paires à afficher
IMG_SIZE  = (20, 20)
CLASS_A   = "gunshot"
CLASS_B   = "fireworks"
# ───────────────────────────────────────────────────────────────────────────────

idx_a = np.where(labels == CLASS_A)[0]
idx_b = np.where(labels == CLASS_B)[0]

# Matrice de similarité : chaque gunshot vs chaque firework
sim_matrix = cosine_similarity(data[idx_a], data[idx_b])  # (n_gunshot, n_fireworks)

# ── Trouver les N_PAIRS paires les plus similaires ────────────────────────────
# On "aplatit" la matrice et on prend les indices des N meilleures valeurs
flat_top = np.argsort(sim_matrix, axis=None)[::-1]

seen_a, seen_b = set(), set()
pairs = []  # liste de (local_idx_a, local_idx_b, similarity)

for flat_idx in flat_top:
    i, j = np.unravel_index(flat_idx, sim_matrix.shape)
    if i not in seen_a and j not in seen_b:
        pairs.append((i, j, sim_matrix[i, j]))
        seen_a.add(i)
        seen_b.add(j)
    if len(pairs) == N_PAIRS:
        break

# ── Affichage : une figure avec N_PAIRS lignes × 2 colonnes ──────────────────
fig, axes = plt.subplots(N_PAIRS, 2, figsize=(5, 2.4 * N_PAIRS))

for row, (local_i, local_j, sim) in enumerate(pairs):
    global_i = idx_a[local_i]
    global_j = idx_b[local_j]

    for col, (g_idx, img_data) in enumerate([(global_i, data[global_i]),
                                              (global_j, data[global_j])]):
        ax = axes[row, col]
        ax.imshow(img_data.reshape(IMG_SIZE).T,
                  aspect="auto", origin="lower", cmap="jet")
        ax.axis("off")
        # Pas de titre → l'observateur ne sait pas quelle classe c'est
        # (optionnel : décommenter pour debug)
        ax.set_title(labels[g_idx].capitalize(), fontsize=14, fontweight="bold")

# Supprimer tout espacement résiduel entre les subplots
# plt.subplots_adjust(wspace=0.02, hspace=0.05)
plt.savefig("gunshot_vs_fireworks_confusable.png",
            bbox_inches="tight", dpi=300, transparent=True)
plt.show()

print("Paires sélectionnées (similarité cosinus) :")
for rank, (li, lj, sim) in enumerate(pairs, 1):
    print(f"  #{rank}  gunshot[{idx_a[li]}]  ↔  fireworks[{idx_b[lj]}]  sim={sim:.4f}")


# x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

# mlp = MLPClassifier(hidden_layer_sizes=(50, 50, 50), max_iter=200, random_state=42)
# mlp.fit(x_train, y_train)

# # 1. Génération de la matrice
# disp = ConfusionMatrixDisplay.from_estimator(
#     mlp,
#     x_test,
#     y_test,
#     display_labels=MLP_halikarnassos_classes,
#     cmap=plt.cm.Blues,
#     normalize=None,
# )

# # 2. Ajustement des axes (Utilisation de .ax_)
# ticks = range(len(MLP_halikarnassos_classes))

# # Correction : disp.ax_ au lieu de disp.ax
# disp.ax_.set_xticks(ticks)
# disp.ax_.set_xticklabels(MLP_halikarnassos_classes, rotation=45, ha='right', rotation_mode='anchor')

# disp.ax_.set_yticks(ticks)
# disp.ax_.set_yticklabels(MLP_halikarnassos_classes)

# # 3. Augmenter la taille des labels des axes
# disp.ax_.set_xlabel('Predicted labels', fontsize=14, fontweight='bold')
# disp.ax_.set_ylabel('True labels', fontsize=14, fontweight='bold')

# # 4. Ajustement automatique pour ne pas couper les labels inclinés
# plt.tight_layout()
# plt.savefig("basic_confusion_matrix.png", transparent=True, dpi=300)
# plt.show()


# from sklearn.metrics.pairwise import cosine_similarity
# import matplotlib.pyplot as plt

# # ─── Paramètres ────────────────────────────────────────────────────────────────
# TOP_K = 2          # nombre de voisins à afficher par tronçonneuse
# N_CHAINSAWS = 10   # nombre de tronçonneuses à analyser
# IMG_SIZE = (20, 20)
# class_to_analyze = "crackling fire"  # classe à analyser (doit être présente dans les labels)
# # ───────────────────────────────────────────────────────────────────────────────

# chainsaw_idx     = np.where(labels == class_to_analyze)[0]
# non_chainsaw_idx = np.where(labels != class_to_analyze)[0]

# query_idx  = chainsaw_idx[:N_CHAINSAWS]
# queries    = data[query_idx]
# candidates = data[non_chainsaw_idx]

# sim_matrix = cosine_similarity(queries, candidates)  # (N_CHAINSAWS, N_non_chainsaw)

# for i, q_idx in enumerate(query_idx):

#     top_local  = np.argsort(sim_matrix[i])[::-1][:TOP_K]
#     top_global = non_chainsaw_idx[top_local]

#     # ── Figure : 1 ligne = query | TOP_K voisins ──────────────────────────────
#     fig, axes = plt.subplots(1, TOP_K + 1, figsize=(3 * (TOP_K + 1), 3.5))
#     # fig.suptitle(f"Crackling fire #{i+1}  (index {q_idx})", fontsize=13, fontweight="bold")

#     # Query
#     ax = axes[0]
#     ax.imshow(queries[i].reshape(IMG_SIZE).T, aspect="auto", origin="lower", cmap="jet")
#     # ax.set_title(f"QUERY\n{class_to_analyze}", fontsize=9, color="crimson", fontweight="bold")
#     ax.set_title(f"{class_to_analyze.capitalize()}", fontsize=14, fontweight="bold")
#     ax.axis("off")

#     # Voisins
#     for rank, (local_i, global_i) in enumerate(zip(top_local, top_global), 1):
#         ax = axes[rank]
#         ax.imshow(data[global_i].reshape(IMG_SIZE).T, aspect="auto", origin="lower", cmap="jet")
#         ax.set_title(
#             f"{labels[global_i].capitalize()}",#f"#{rank}  sim={sim_matrix[i][local_i]:.3f}\n{labels[global_i]}",
#             fontsize=14,
#             fontweight="bold",
#         )
#         ax.axis("off")

#     plt.tight_layout()
#     plt.show()





















# # Supposons que `data` est un tableau numpy de forme (n_samples, height, width) ou (n_samples, height, width, channels)
# # et que `labels_encoded` contient les étiquettes encodées pour chaque échantillon.
# for time in range(10):
#     # Initialiser une figure pour afficher les spectrogrammes
#     plt.figure(figsize=(15, 8))

#     # Parcourir chaque classe
#     for i, class_name in enumerate(MLP_halikarnassos_classes):
#         # Trouver les indices des échantillons appartenant à la classe actuelle
#         class_indices = np.where(labels_encoded == i)[0]

#         # Sélectionner un indice aléatoire parmi les indices de la classe
#         random_index = random.choice(class_indices)

#         # Récupérer le spectrogramme correspondant
#         spectrogram = data[random_index].reshape(20, 20).T  # Assurez-vous que la forme correspond à celle de vos spectrogrammes

#         # Ajouter un sous-graphe pour cette classe
#         plt.subplot(2, 3, i + 1)  # 2 lignes, 3 colonnes, position i+1
#         plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='jet')
#         plt.title(f"{class_name}".capitalize(), fontsize=14, fontweight='bold')
#         plt.axis('off')

#     # Ajuster la disposition et afficher
#     plt.tight_layout()
#     plt.savefig(f"sample_spectrograms_{time}.png", transparent=True, dpi=300)
#     plt.close()