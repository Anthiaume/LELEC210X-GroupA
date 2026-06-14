import os
import subprocess
from time import time, sleep
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pickle

# Import MLP classifier, KNN classifier and train_test_split from sklearn

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# from classification.src.classification.utils.utils import accuracy

warnings.filterwarnings("ignore")

def get_class_str(string : str):
    if "background" in string.lower():
        return "background"
    elif "chainsaw" in string.lower():
        return "chainsaw"
    elif "fire" in string.lower() and "fireworks" not in string.lower():
        return "fire"
    elif "fireworks" in string.lower():
        return "fireworks"
    elif "gunshot" in string.lower():
        return "gunshot"
    elif "general" in string.lower():
        return "general"
    else:
        raise f"Unexpected class in string: {string}"
    
def get_class_int(string : str):
    if "background" in string.lower():
        return 0
    elif "chainsaw" in string.lower():
        return 1
    elif "fire" in string.lower() and "fireworks" not in string.lower():
        return 2
    elif "fireworks" in string.lower():
        return 3
    elif "gunshot" in string.lower():
        return 4
    elif "general" in string.lower():
        return 5
    else:
        raise f"Unexpected class in string: {string}"

def rule_majority_vote(predictions):
    # Tri alphabétique pour garantir un résultat déterministe en cas d'ex-æquo
    candidates = sorted(set(predictions))
    winner = max(candidates, key=predictions.count)
    return winner, "rule_majority_vote"

def rule_grand_mere(models, predictions):
    # background AF, chainsaw AF, fire AF
    candidates = sorted(predictions)
    if candidates.count("fireworks") >= 9:
        return "fireworks", "rule_grand_mere"
    else:
        if "fireworks" in candidates:
            candidates =  [elem for elem in candidates if elem != "fireworks"]
        winner = max(candidates, key=candidates.count)
        return winner, "rule_grand_mere"

def compute_precision_table(confusion_matrices, models):
    """
    Pour chaque modèle et chaque classe prédite, calcule :
    P(prédiction correcte | modèle m prédit classe c)
    = CM[c, c] / sum_ligne(CM[:, c])
    Agrège les 3 niveaux de volume (25%, 50%, 100%).
    """
    classes = ["background", "chainsaw", "fire", "fireworks", "gunshot"]
    precision = {}
    for model in models:
        cm_total = sum(
            confusion_matrices[model][l].astype(float)
            for l in ["data_25", "data_50", "data_100"]
        )
        precision[model] = {}
        for j, cls in enumerate(classes):
            col_sum = cm_total[:, j].sum()
            precision[model][cls] = cm_total[j, j] / col_sum if col_sum > 0 else 0.5
    return precision

def rule_weighted_precision(models_used, predictions, precision_table):
    """
    Chaque vote est pondéré par la précision empirique du modèle
    pour la classe qu'il a prédite.

    Exemple :
      - background_v9_AF prédit "background"  → +0.976
      - fire_v26_HF prédit "fireworks"        → +0.384 seulement
      → fireworks perd naturellement face aux vrais signaux
    """
    classes = ["background", "chainsaw", "fire", "fireworks", "gunshot"]
    scores = {c: 0.0 for c in classes}

    for model, pred in zip(models_used, predictions):
        weight = precision_table.get(model, {}).get(pred, 0.5)
        scores[pred] += weight

    winner = max(scores, key=scores.get)
    return winner, "rule_weighted_precision"

filenameGun = ["results_uart_gunshot_25.txt", "results_uart_gunshot_50.txt", "results_uart_gunshot_100.txt"]
filenameBac = ["results_uart_background_25.txt", "results_uart_background_50.txt", "results_uart_background_100.txt"]
filenameCha = ["results_uart_chainsaw_25.txt", "results_uart_chainsaw_50.txt", "results_uart_chainsaw_100.txt"]
filenameFir = ["results_uart_fire_25.txt", "results_uart_fire_50.txt", "results_uart_fire_100.txt"]
filenameWor = ["results_uart_fireworks_25.txt", "results_uart_fireworks_50.txt", "results_uart_fireworks_100.txt"]

filnames = {
    "gunshot": filenameGun,
    "background": filenameBac,
    "chainsaw": filenameCha,
    "fire": filenameFir,
    "fireworks": filenameWor
}

# labels_intensity = ["25%", "50%", "100%"]
fontsize = 12

# Read the data
raw_data = {
    "gunshot_data": [open(file, 'r').readlines() for file in filenameGun],
    "background_data": [open(file, 'r').readlines() for file in filenameBac],
    "chainsaw_data": [open(file, 'r').readlines() for file in filenameCha],
    "fire_data": [open(file, 'r').readlines() for file in filenameFir],
    "fireworks_data": [open(file, 'r').readlines() for file in filenameWor]
}

refined_data = {
    "gunshot_data": {
        "data_25": [],
        "data_50": [],
        "data_100": []
    },
    "background_data": {
        "data_25": [],
        "data_50": [],
        "data_100": []
    },
    "chainsaw_data": {
        "data_25": [],
        "data_50": [],
        "data_100": []
    },
    "fire_data": {
        "data_25": [],
        "data_50": [],
        "data_100": []
    },
    "fireworks_data": {
        "data_25": [],
        "data_50": [],
        "data_100": []
    }
}

# Extract model names and predictions
for data_type in raw_data.keys():
    for file_data in range(len(raw_data[data_type])):
        if file_data == 0:
            loudness = "data_25"
        elif file_data == 1:
            loudness = "data_50"
        elif file_data == 2:
            loudness = "data_100"
        else:
            raise "Unexpected file index, expected 0, 1 or 2"
        
        refined_data[data_type][loudness].append([line.strip().split(" ")[1:][0] for line in raw_data[data_type][file_data] if line.strip() not in ("New predictions :", "")])
        refined_data[data_type][loudness].append([line.strip().split(" ")[-1][1:] for line in raw_data[data_type][file_data] if line.strip() not in ("New predictions :", "")])

models = [str(elem) for elem in np.unique(refined_data["gunshot_data"]["data_25"][0])]
models.sort(key=lambda x: x.split("_")[0]+x.split("_")[2])

claude_technique = False
if claude_technique:
    # ── Calcul de la table de précision ──────────────────────────────────────────
    # On reconstruit les CM individuelles avant la boucle d'évaluation
    confusion_matrices = {
        model: {
            "data_25":  np.zeros((5, 5), dtype=int),
            "data_50":  np.zeros((5, 5), dtype=int),
            "data_100": np.zeros((5, 5), dtype=int),
        }
        for model in models
    }

    label_map = {
        "background_data": 0, "chainsaw_data": 1, "fire_data": 2,
        "fireworks_data": 3, "gunshot_data": 4,
    }
    pred_map = {"background": 0, "chainsaw": 1, "fire": 2, "fireworks": 3, "gunshot": 4}

    for real_data in refined_data.keys():
        real_label = label_map[real_data]
        for loudness in refined_data[real_data].keys():
            for i, (m, p) in enumerate(zip(
                refined_data[real_data][loudness][0],
                refined_data[real_data][loudness][1]
            )):
                if m in confusion_matrices and p in pred_map:
                    confusion_matrices[m][loudness][real_label, pred_map[p]] += 1

    PRECISION_TABLE = compute_precision_table(confusion_matrices, models)

    # ── Boucle d'évaluation principale ───────────────────────────────────────────
    well_classified = 0
    total = 0
    confusion_matrix_decision_rule = np.zeros((5, 5), dtype=float)

    for real_data in refined_data.keys():
        real_label = get_class_int(real_data)
        for loudness in refined_data[real_data].keys():
            for i in range(len(refined_data[real_data][loudness][0]) // 12):
                models_used  = refined_data[real_data][loudness][0][i*12:(i+1)*12]
                predictions  = refined_data[real_data][loudness][1][i*12:(i+1)*12]

                predicted_label, rule = rule_weighted_precision(
                    models_used, predictions, PRECISION_TABLE
                )

                if predicted_label == get_class_str(real_data):
                    well_classified += 1
                total += 1
                confusion_matrix_decision_rule[real_label, get_class_int(predicted_label)] += 1

# Evaluate accuracy by majority vote for each model and loudness level

well_classified = 0
total = 0

confusion_matrix_decision_rule = np.zeros((5, 5), dtype=float)

inputs = []
outputs = []

# for real_data in refined_data.keys():

#     real_label = get_class_int(real_data)

#     for loudness in refined_data[real_data].keys():
#         for i in range(len(refined_data[real_data][loudness][0])//12):

#             models_used = refined_data[real_data][loudness][0][i*12:(i+1)*12]
#             predictions = refined_data[real_data][loudness][1][i*12:(i+1)*12]

#             inputs.append(predictions)
#             outputs.append(get_class_str(real_data))

#             # predicted_label, rule = rule_majority_vote(predictions)
#             predicted_label, rule = rule_grand_mere(models_used, predictions)
#             model = pickle.load(open("mlp_model.pkl", "rb"))
#             classes = ["background", "chainsaw", "fire", "fireworks", "gunshot"]
#             predicted_label = classes[model.predict(np.array([get_class_int(pred) for pred in predictions]).reshape(1, -1))[0]]

#             if predicted_label == get_class_str(real_data):
#                 well_classified += 1
#             total += 1

#             predicted_label_int = get_class_int(predicted_label)

#             confusion_matrix_decision_rule[real_label, predicted_label_int] += 1

# print(f"Overall accuracy: {well_classified/total:.1%} ({well_classified}/{total})")

# mlp = MLPClassifier(random_state=42, max_iter=1000, learning_rate="adaptive", early_stopping=True, hidden_layer_sizes=(100, 100, 100), activation="relu", solver="adam")
# # Encode string labels to integers for MLP

# inputs_float = np.array([[get_class_int(pred) for pred in sample] for sample in inputs], dtype=float)
# outputs_int = np.array([get_class_int(label) for label in outputs])

# print(inputs_float.shape, outputs_int.shape)

# x_train, x_test, y_train, y_test = train_test_split(inputs_float, outputs_int, test_size=0.2, random_state=42, stratify=outputs_int, shuffle=True)


# mlp.fit(inputs_float, outputs_int)
# score_mlp = mlp.score(x_test, y_test)
# print(f"MLP accuracy: {score_mlp:.1%} ({int(score_mlp*len(y_test))}/{len(y_test)})")


# pickle.dump(mlp, open("mlp_supermodel.pkl", "wb"))

# # Display Normalized Confusion Matrix
# classes = ["background", "chainsaw", "fire", "fireworks", "gunshot"]
# cm_normalized = confusion_matrix_decision_rule / np.sum(confusion_matrix_decision_rule) * 100
# disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=classes)
# disp.plot(cmap=plt.cm.Blues, colorbar=False)
# disp.ax_.set_xticklabels(classes, rotation=45, ha='right', rotation_mode='anchor', fontsize=fontsize)
# disp.ax_.set_yticklabels(classes, fontsize=fontsize)
# disp.ax_.set_xlabel("Predicted Label", fontsize=fontsize*1.2)
# disp.ax_.set_ylabel("True Label", fontsize=fontsize*1.2)
# disp.ax_.set_title(f"{rule}\nAccuracy: {well_classified/total:.1%}", fontsize=fontsize*1.5)
# plt.tight_layout()
# if 1:
#     plt.savefig(f"CONFUSION_MATRIX_Live_final_models_{rule}.pdf", bbox_inches='tight')
#     plt.savefig(f"temp.pdf", bbox_inches='tight')
#     subprocess.Popen(["temp.pdf"],shell=True)
#     sleep(1)
#     os.remove(f"temp.pdf")


to_plot = True
if to_plot:
    # Three confusion matrices per model (one for each loudness level)

    # Generate dict to store confusion matrices with model names as keys
    confusion_matrices = {model: {"data_25": np.zeros((5, 5), dtype=int), "data_50": np.zeros((5, 5), dtype=int), "data_100": np.zeros((5, 5), dtype=int)} for model in models}

    # Fill confusion matrices
    for real_data in refined_data.keys():

        for loudness in refined_data[real_data].keys():
            
            if len(refined_data[real_data][loudness][0]) != len(refined_data[real_data][loudness][1]):
                raise "Unexpected data length, expected same number of model labels and predicted labels"
            
            for i in range(len(refined_data[real_data][loudness][0])):
                
                if refined_data[real_data][loudness][0][i] not in models:
                    raise "Unexpected model label, expected one of the following: " + ", ".join(models)
                
                if real_data == "background_data":
                    real_label = 0
                elif real_data == "chainsaw_data":
                    real_label = 1
                elif real_data == "fire_data":
                    real_label = 2
                elif real_data == "fireworks_data":
                    real_label = 3
                elif real_data == "gunshot_data":
                    real_label = 4
                else:
                    raise "Unexpected real data type, expected background_data, chainsaw_data, fire_data, fireworks_data or gunshot_data"
                
                predicted_label = refined_data[real_data][loudness][1][i]

                if predicted_label == "background":
                    predicted_label = 0
                elif predicted_label == "chainsaw":
                    predicted_label = 1
                elif predicted_label == "fire":
                    predicted_label = 2
                elif predicted_label == "fireworks":
                    predicted_label = 3
                elif predicted_label == "gunshot":
                    predicted_label = 4
                else:
                    raise f"Unexpected predicted label, expected background, chainsaw, fire, fireworks or gunshot, got {predicted_label}."

                

                model_label = refined_data[real_data][loudness][0][i]

                confusion_matrices[model_label][loudness][real_label, predicted_label] += 1

    # Print confusion matrices
    for model in confusion_matrices.keys():
        print(f"Confusion matrices for model {model}:")
        for loudness in confusion_matrices[model].keys():
            print(f"Confusion matrix for loudness level {loudness}:")
            print(confusion_matrices[model][loudness])

    classes = ["background", "chainsaw", "fire", "fireworks", "gunshot"]
    def accuracy(cm):
        return np.trace(cm) / np.sum(cm)
    for plot in range(3):

        # Display Normalized Confusion Matrices
        n_plots = 12
        factor = 1
        fig, axes = plt.subplots(4, 3, figsize=(4*4*factor, 6*4*factor))


        for model in range(4):

            for subplot in range(3):

                cm = confusion_matrices[models[model + plot*4]][["data_25", "data_50", "data_100"][subplot]].astype(float)

                cm_normalized = cm / np.sum(cm) * 100


                disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=classes)
                disp.plot(cmap=plt.cm.Blues, ax=axes[model, subplot], colorbar=False)
                disp.ax_.set_xticklabels(classes, rotation=45, ha='right', rotation_mode='anchor', fontsize=fontsize)
                disp.ax_.set_yticklabels(classes, fontsize=fontsize)
                disp.ax_.set_xlabel("Predicted Label", fontsize=fontsize*1.2)
                disp.ax_.set_ylabel("True Label", fontsize=fontsize*1.2)
                title = ""
                for cls in classes + ["general"]:
                    if cls in models[model + plot*4].lower() and "fire" not in cls:
                        title = cls.capitalize()
                        break
                    elif cls in models[model + plot*4].lower():
                        
                        title = "Fire" if "fireworks" not in models[model + plot*4].lower() else "Fireworks"
                        print("title =", title)
                        break
                title += " " + "HF" if "hf" in models[model + plot*4].lower() else " AF"
                title += f"\nAccuracy: {accuracy(cm):.1%}"
                title += f"\nLoudness: {['25%', '50%', '100%'][subplot]}"
                axes[model, subplot].set_title(title, fontsize=fontsize*1.5)

        # for j in range(model+1, len(axes)):
        #     fig.delaxes(axes[j])
        plt.tight_layout()
        if 1:
            if plot == 0:
                classes_plot = "background_chainsaw"
                plt.savefig(f"CONFUSION_MATRICES_Live_final_models_{classes_plot}.pdf", bbox_inches='tight')
            elif plot == 1:
                classes_plot = "fire_fireworks"
                plt.savefig(f"CONFUSION_MATRICES_Live_final_models_{classes_plot}.pdf", bbox_inches='tight')
            elif plot == 2:
                classes_plot = "gunshot_general"
                plt.savefig(f"CONFUSION_MATRICES_Live_final_models_{classes_plot}.pdf", bbox_inches='tight')
        if 1:
            plt.savefig(f"temp.pdf", bbox_inches='tight')
            subprocess.Popen(["temp.pdf"],shell=True)
            sleep(1)
            os.remove(f"temp.pdf")
        plt.close()