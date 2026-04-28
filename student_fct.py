from asyncio import subprocess
import os
from os import listdir
from os.path import isfile, join
import pickle
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import time
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import subprocess
import scipy, librosa

def load_models():
    models = []
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classification", "data", "models")
    # Load all models
    for file in os.listdir(path):
        if file.endswith(".pkl"):
            model = pickle.load(open(os.path.join(path, file), "rb"))
            models.append(model)
    return models

def process_data_for_MLP(x_data, params):

    x_data_reshaped = x_data.reshape(1, 400)

    x_data_HF = keep_frequencies(data=x_data_reshaped, n_melvecs_to_keep=params["keeped_frequencies"])

    x_data_HF_normalized = x_data_HF / (np.linalg.norm(x_data_HF, axis=1, keepdims=True) + 1e-16)  # Add a small value to avoid division by zero

    if params["attenuation_dB_range"] is not None:
        x_data_HF_normalized, _ = add_background(data_normalized=x_data_HF_normalized, labels=np.zeros(x_data_HF_normalized.shape[0], dtype=int), attenuation_dB_range=params["attenuation_dB_range"])

    if params["transform_mean_std"]:
        x_data_HF_normalized = transform_melspecgram_to_mean_std_data(x_data_HF_normalized, n_freq=params["n_freq"])

    x_data_HF_normalized = x_data_HF_normalized**params["exposant"]

    return x_data_HF_normalized

def process_data_for_MLP_for_test(x_data, params):

    x_data_HF = keep_frequencies(data=x_data, n_melvecs_to_keep=params["keeped_frequencies"])

    x_data_HF_normalized = x_data_HF / (np.linalg.norm(x_data_HF, axis=1, keepdims=True) + 1e-16)  # Add a small value to avoid division by zero

    if params["attenuation_dB_range"] is not None:
        x_data_HF_normalized, _ = add_background(data_normalized=x_data_HF_normalized, labels=np.zeros(x_data_HF_normalized.shape[0], dtype=int), attenuation_dB_range=params["attenuation_dB_range"])

    if params["transform_mean_std"]:
        x_data_HF_normalized = transform_melspecgram_to_mean_std_data(x_data_HF_normalized, n_freq=params["n_freq"])

    x_data_HF_normalized = x_data_HF_normalized**params["exposant"]

    return x_data_HF_normalized

def load_data(records, classes=["background", "chainsaw", "crackling fire", "fireworks", "gunshot"]):
    """
    Inputs:
    - records : list of tuples (mcu, local, speaker) specifying the data to load. For example:
        Example : records = [("mcu13", "vinikot", "JBL Flip 5 - Auguste - spec_20_20"),
                             ("mcu13", "fisher", "local speakers - spec_20_20")       ]
    Output:
    - data: numpy array of melspectrograms
    - labels: numpy array of labels (e.g. "chainsaw", "crackling fire", "fireworks", "gunshot", "background")
    - labels_encoded: numpy array of labels encodés (ex: 0, 1, 2, 3, 4)
    - classes: numpy array des classes (ex: ["chainsaw", "crackling fire", "fireworks", "gunshot", "background"])
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
        files_in_directory = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith(".pkl") and f.startswith(tuple(classes)))]
        for file in files_in_directory:
            with open (os.path.join(mypath, file), "rb") as f:
                data.append(pickle.load(f))
                labels.append(file.split("_")[0])
                
    data = np.array(data)
    labels = np.array(labels)

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    classes = le.classes_

    return data, labels, labels_encoded, classes

def load_compacted_data(records, n_samples_per_new_sample):
    data, labels = load_data(records)
    n_labels = len(np.unique(labels))
    n_samples_per_class = (data.shape[0] // n_labels) // n_samples_per_new_sample
    new_data = [""] * (n_labels * n_samples_per_class)
    new_labels = [""] * (n_labels * n_samples_per_class)

    for i in range(n_labels):
        class_data = data[labels == np.unique(labels)[i]]
        class_labels = labels[labels == np.unique(labels)[i]]
        for j in range(n_samples_per_class):
            start_idx = j * n_samples_per_new_sample
            end_idx = (j + 1) * n_samples_per_new_sample
            new_data[i * n_samples_per_class + j] = np.concatenate(class_data[start_idx:end_idx], axis=0)
            new_labels[i * n_samples_per_class + j] = class_labels[start_idx]

    new_data = np.array(new_data)
    new_labels = np.array(new_labels)

    return new_data, new_labels
      
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

def save_confusion_matrix_from_array(confusion_matrix=None, filename="CONFUSION_MATRIX.pdf", show=True, labels=None):
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
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()

def transform_melspecgram_to_mean_std_data(data_normalized, n_freq=20):
    """
    Description: Cette fonction prend en entrée des melspectrogrammes normalisés (norme L2 de 1 pour chaque échantillon) et les transforme en données compactées où pour chaque échantillon, on a les moyennes et les écarts-types de chaque melvector (20 moyennes + 20 écarts-types = 40).
    Inputs:
        data_normalized: numpy array de shape (n_samples, 400) correspondant à des melspectrogrammes de 20 melvectors de longueur 20 (20*20=400) et normalisés (norme L2 de 1 pour chaque échantillon)
        n_freq: nombre de fréquences (melvectors)
    Output:
        data_compacted: numpy array de shape (n_samples, 40) correspondant à des données compactées où pour chaque échantillon, on a les moyennes et les écarts-types de chaque melvector (20 moyennes + 20 écarts-types = 40)
    """
    if data_normalized.shape[1] != 20*n_freq:
        raise ValueError(f"Expected data with shape (n_samples, {20*n_freq}) corresponding to {n_freq} melvectors of length 20 ({n_freq}*20={20*n_freq})")
    if np.linalg.norm(data_normalized, axis=1).mean() < 1 - 1e-2 or np.linalg.norm(data_normalized, axis=1).mean() > 1 + 1e-2:
        raise ValueError("Expected data to be normalized with L2 norm of 1 for each sample (mean norm: {:.4f})".format(np.linalg.norm(data_normalized, axis=1).mean()))
    
    data_normalized_reshaped_transposed = np.transpose(data_normalized.reshape(data_normalized.shape[0], 20, n_freq), (0, 2, 1))

    data_normalized_freq_mean = np.mean(data_normalized_reshaped_transposed, axis=2, keepdims=True).reshape(data_normalized_reshaped_transposed.shape[0], n_freq)
    data_normalized_freq_std  = np.var(data_normalized_reshaped_transposed,  axis=2, keepdims=True).reshape(data_normalized_reshaped_transposed.shape[0], n_freq)**0.5

    data_compacted = np.concatenate((data_normalized_freq_mean, data_normalized_freq_std), axis=1)
    return data_compacted

def plot_specgram(
    specgram,
    ax,
    is_mel=False,
    title=None,
    xlabel="Time [s]",
    ylabel="Frequency [Hz]",
    cmap="jet",
    cb=True,
    tf=None,
    invert=True,
):
    """
    Plot a spectrogram (2D matrix) in a chosen axis of a figure.
    Inputs:
        - specgram = spectrogram (2D array)
        - ax       = current axis in figure
        - title
        - xlabel
        - ylabel
        - cmap
        - cb       = show colorbar if True
        - tf       = final time in xaxis of specgram
    """
    if tf is None:
        tf = specgram.shape[1]

    if is_mel:
        ylabel = "Frequency [Mel]"
        im = ax.imshow(
            specgram, cmap=cmap, aspect="auto", extent=[0, tf, specgram.shape[0], 0]
        )
    else:
        im = ax.imshow(
            specgram,
            cmap=cmap,
            aspect="auto",
            extent=[0, tf, int(specgram.size / tf), 0],
        )
    if invert:
        ax.invert_yaxis()
    fig = plt.gcf()
    if cb:
        fig.colorbar(im, ax=ax)
    # cbar.set_label('log scale', rotation=270)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return None

def add_background(data_normalized, labels, attenuation_dB_range=(-20, -15)):
    """"
    Description: Cette fonction ajoute un bruit de fond (background) à chaque échantillon utile du dataset.
                 Très important de passer cette fonction sur des échantillons normalisés (ex: MEL spectrogrammes) pour que l'atténuation en dB soit cohérente.
    Inputs:
        - data: numpy array de forme (n_samples, n_features) contenant les échantillons du dataset
        - labels: numpy array de forme (n_samples,) contenant les labels correspondants à chaque échantillon
        - attenuation_dB_range: tuple (min_dB, max_dB) spécifiant la plage de valeurs d'atténuation en décibels à appliquer au bruit de fond avant de l'ajouter aux échantillons utiles. Par exemple, (-20, -15) signifie que le bruit de fond sera atténué entre 20 dB et 15 dB avant d'être ajouté.
    Outputs:
        - new_data: numpy array de forme (n_samples, n_features) contenant les échantillons du dataset après l'ajout du bruit de fond
        - new_labels: numpy array de forme (n_samples,) contenant les labels correspondants à chaque échantillon dans new_data (identiques à labels)
    """
    
    def plot3(original, noise, combined):
        N_MELVECS = 20
        MELVEC_LENGTH = 20

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot original
        plot_specgram(
            original.reshape((N_MELVECS, MELVEC_LENGTH)).T,
            ax=axes[0],
            is_mel=True,
            title=f"MEL Spectrogram : original",
            xlabel="Mel vector",
        )

        # Plot noise        
        plot_specgram(
            noise.reshape((N_MELVECS, MELVEC_LENGTH)).T,
            ax=axes[1],
            is_mel=True,
            title=f"MEL Spectrogram : background noise",
            xlabel="Mel vector",
        )

        # Plot combined
        plot_specgram(
            combined.reshape((N_MELVECS, MELVEC_LENGTH)).T,
            ax=axes[2],
            is_mel=True,
            title=f"MEL Spectrogram : combined",
            xlabel="Mel vector",
        )
        plt.tight_layout()
        plt.show()  
        plt.close()

    background_data = data_normalized[labels == 0]
    background_labels = labels[labels == 0]
    usefull_data = data_normalized[labels != 0]
    usefull_labels = labels[labels != 0]

    for sample in range(usefull_data.shape[0]):
        background_index = random.randint(0, background_data.shape[0] - 1)

        attenuation_dB = np.random.uniform(*attenuation_dB_range)
        attenuation_factor = 10 ** (attenuation_dB / 20)

        # copy_to_plot = usefull_data[sample,:].copy()

        new_data = usefull_data[sample,:] + (attenuation_factor * background_data[background_index,:].astype(np.float64)).astype(usefull_data.dtype)
        new_data_normalized = new_data / np.linalg.norm(new_data)
        usefull_data[sample,:] = new_data_normalized

        # plot3(copy_to_plot, background_data[background_index,:], usefull_data[sample,:])

    return np.concatenate((usefull_data, background_data)), np.concatenate((usefull_labels, background_labels))

def suppress_low_frequencies(data, n_melvecs_to_suppress=10):
    """
    Description: Cette fonction supprime les n_melvecs_to_suppress premiers melvecs (les plus basses fréquences) de chaque échantillon du dataset.
    Inputs:
        - data: numpy array de forme (n_samples, n_features) contenant les échantillons du dataset
        - n_melvecs_to_suppress: int spécifiant le nombre de melvalues à supprimer pour chaque melvector. Par exemple, si n_melvecs_to_suppress=10, cela signifie que les 10 premières melvalues de chaque melvector seront supprimées, ce qui correspond à la suppression des fréquences les plus basses. Note: dans un melvector de longueur 20, les 10 premières melvalues correspondent aux fréquences les plus basses et les 10 dernières melvalues correspondent aux fréquences les plus hautes.
    Outputs:
        - new_data: numpy array de forme (n_samples, n_features - n_melvecs_to_suppress) contenant les échantillons du dataset après la suppression des melvecs
   
    Description of melspectrograms frequencies location:
    - Melspectrograms saved in shape (400,) correspond to 20 melvectors of length 20 (20*20=400)
    - The first 10 mel values of each melvector correspond to the lowest frequencies
    - The last 10 mel values of each melvector correspond to the highest frequencies
    Example:
    >>> melvec.shape
    (400,)
    >>> print("Low frequencies:")
    >>> print(melvec[0:10], melvec[20:30], melvec[40:50], melvec[60:70], melvec[80:90], melvec[100:110], melvec[120:130], melvec[140:150], melvec[160:170], melvec[180:190], ...)
    
    >>> print("High frequencies:")
    >>> print(melvec[10:20], melvec[30:40], melvec[50:60], melvec[70:80], melvec[90:100], melvec[110:120], melvec[130:140], melvec[150:160], melvec[170:180], melvec[190:200], ...)
    
    --> Goal: suppress and delete the low frequencies (the first 10 mel values of each melvector) and keep only the high frequencies (the last 10 mel values of each melvector)
    """

    # Masque pour chaque ligne
    mask = np.ones((data.shape[0], 400), dtype=bool)
    mask[:, np.r_[0:400:20][:, None] + np.arange(n_melvecs_to_suppress)] = False

    # Résultat : tableau de taille (x, 200)
    data = data[mask].reshape(data.shape[0], -1)
    return data

def keep_frequencies(data, n_melvecs_to_keep=(0,19)):

    # Masque pour chaque ligne
    mask = np.zeros(data.shape, dtype=bool)

    mask[:, np.r_[0:400:20][:, None] + np.arange(n_melvecs_to_keep[0], n_melvecs_to_keep[1]+1)] = True

    # Résultat : tableau de taille (x, 200)
    data = data[mask].reshape(data.shape[0], -1)
    return data

def train_and_save_model(params, data_construction, data, labels_encoded, MLP_classes, class_name, bool_save_model=True):

    # Garder seulement les fréquences souhaitées
    data_HF = keep_frequencies(data=data, n_melvecs_to_keep=data_construction["keeped_frequencies"])

    # Normaliser
    data_HF_normalized = data_HF / (np.linalg.norm(data_HF, axis=1, keepdims=True) + 1e-16)  # Add a small value to avoid division by zero

    # Rajouter le bruit
    bool_add_background = data_construction["attenuation_dB_range"] is not None
    if bool_add_background:
        data_HF_normalized, labels_encoded = add_background(data_normalized=data_HF_normalized, labels=labels_encoded, attenuation_dB_range=data_construction["attenuation_dB_range"])

    # Transformer les melspectrogrammes en données compactées (moyennes et écarts-types de chaque melvector) si demandé
    if data_construction["transform_mean_std"]:
        data_HF_normalized = transform_melspecgram_to_mean_std_data(data_HF_normalized, n_freq=data_construction["n_freq"])

    # Appliquer une transformation non linéaire (exposant)
    data_HF_normalized = data_HF_normalized**data_construction["exposant"]

    # Diviser le dataset en train, val, test
    x_train, x_test, y_train, y_test = train_test_split(data_HF_normalized, labels_encoded, test_size=0.2, random_state=42, shuffle=True, stratify=labels_encoded)
    x_train, x_val, y_train, y_val   = train_test_split(x_train, y_train, test_size=0.2, random_state=42, shuffle=True, stratify=y_train)

    # Training the model
    duration = time.time()
    mlp = TorchMLP(hidden_layers_sizes=params["hidden_layers_sizes"],
                    activation=params["activation"],
                    IO=params["IO"], 
                    num_epochs=params["num_epochs"],
                    batch_size=params["batch_size"],
                    learning_rate=params["learning_rate"],
                    dropout_rate=params["dropout_rate"],
                    class_weights=params["class_weights"],
                    x_val=x_val, y_val=y_val,
                    plot_loss=params["plot_loss"],
                    verbose=params["verbose"],
                    loss_filename=params["loss_filename"])
    train_losses, val_losses = mlp.fit(x_train, y_train)
    duration = time.time() - duration
    print(f"Training time: {duration:.2f} seconds")

    # Print performances and save confusion matrix
    train_acc = mlp.score(x_train, y_train)
    val_acc  = mlp.score(x_val, y_val)
    test_acc  = mlp.score(x_test, y_test)
    diff_acc = train_acc - test_acc
    print(f"Accuracy train : {train_acc:.4f}")
    print(f"Accuracy val   : {val_acc:.4f}")
    print(f"Accuracy test  : {test_acc:.4f}" )
    print(f"Difference train - test accuracy: {diff_acc:.4f}")

    # Sert à Visualiser les probas des FP, TP, FN
    # mlp.save_proba_class_x(x_val, y_val, class_idx=3, class_name="fireworks")

    # Montre la matrice de confusion
    mlp.save_confusion_matrix(x_val, y_val, class_names=MLP_classes, show=True, filename=None)
    mlp.save_confusion_matrix(x_test, y_test, class_names=MLP_classes, show=True, filename=None)

    # Save model results
    results = ModelResults(model_name = params["model_name"],
                            model=mlp,
                            accuracy={"train": train_acc, "test": test_acc},
                            confusion_matrix=mlp.get_confusion_matrix(x_test, y_test),
                            epochs=params["num_epochs"],
                            classes=MLP_classes,
                            loss_curves={"train": train_losses, "val": val_losses},
                            params=params|data_construction
                            )
    results.save(f"results_{params['model_name']}.pkl")

    if bool_save_model:
        # Save model on all the dataset
        duration = time.time()
        mlp = TorchMLP(hidden_layers_sizes=params["hidden_layers_sizes"],
                    activation=params["activation"],
                    IO=params["IO"],
                    num_epochs=params["num_epochs"],
                    batch_size=params["batch_size"],
                    learning_rate=params["learning_rate"],
                    dropout_rate=params["dropout_rate"],
                    class_weights=params["class_weights"],
                    x_val=None, 
                    y_val=None,
                    plot_loss=True,
                    verbose=True,
                    loss_filename=None)
        train_losses, _ = mlp.fit(data_HF_normalized, labels_encoded)
        duration = time.time() - duration
        print(f"Training time on all the dataset: {duration//3600}h {duration//60%60:.2f}min {duration%60:.2f}s")

        results = ModelResults(model_name = params["model_name"],
                                model=mlp,
                                accuracy={"train": train_acc, "test": test_acc},
                                confusion_matrix=None,
                                classes=MLP_classes,
                                epochs=params["num_epochs"],
                                loss_curves={"train": train_losses, "val": None},
                                params=params|data_construction
                                )
        results.save(f"model_{params['model_name']}.pkl")

        print(f"The model's classes are: {MLP_classes}")






            



class ModelResults:
    def __init__(self, model_name=None, model=None, accuracy=None, confusion_matrix=None, classes=None, epochs=None, loss_curves=None, params=None):
        self.model_name = model_name
        self.model = model
        self.confusion_matrix = confusion_matrix
        self.classes = classes
        self.epochs = epochs
        self.accuracy = accuracy
        self.loss_curves = loss_curves
        self.params = params

    def save(self, filename):
        dict_to_save = {
            "model_name": self.model_name,
            "model": self.model,
            "accuracy": self.accuracy,
            "confusion_matrix": self.confusion_matrix,
            "classes": self.classes,
            "epochs": self.epochs,
            "loss_curves": self.loss_curves,
            "params": self.params
        }
        pickle.dump(dict_to_save, open(filename, "wb"))
    
    def load_and_display(self, filenames, savename=None, show=True):
        data = []
        for filename in filenames:
            with open(filename, "rb") as f:
                data.append(pickle.load(f))
        
        # Display Loss curves
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["steelblue", "tomato", "green", "orange", "purple", "cyan", "magenta", "yellow", "brown", "pink", "gray", "olive", "navy", "teal", "maroon", "lime", "coral", "gold", "indigo", "violet"]
        for d in data:
            color = colors.pop(0)
            if d["loss_curves"] is not None and "train" in d["loss_curves"]:
                ax.plot(np.arange(1, len(d["loss_curves"]["train"]) + 1), d["loss_curves"]["train"], label=f"{d['model_name']} Train Loss", color=color)
            if d["loss_curves"] is not None and "test" in d["loss_curves"]:
                ax.plot(np.arange(1, len(d["loss_curves"]["test"]) + 1), d["loss_curves"]["test"], label=f"{d['model_name']} Test Loss", linestyle='--', color=color)
            if d["loss_curves"] is not None and "val" in d["loss_curves"]:
                ax.plot(np.arange(1, len(d["loss_curves"]["val"]) + 1), d["loss_curves"]["val"], label=f"{d['model_name']} Test Loss", linestyle='--', color=color)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Curves")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        if savename:
            plt.savefig(f"LOSS_CURVES_COMPARISON_{savename}.pdf", bbox_inches='tight')
        if show:
            plt.savefig(f"temp.pdf", bbox_inches='tight')
            subprocess.Popen(["temp.pdf"],shell=True)
            time.sleep(1)
            os.remove(f"temp.pdf")
        plt.close()

        # Display Normalized Confusion Matrices
        n_plots = len(data)
        square = np.arange(1,10)**2
        n_axes = int(n_plots**0.5) if n_plots in square else int(np.ceil(np.sqrt(n_plots)))
        fig, axes = plt.subplots(n_axes, n_axes, figsize=(6*n_axes, 5*n_axes))
        axes = axes.flatten() if n_axes > 1 else [axes]
        for i, d in enumerate(data):
            cm = d["confusion_matrix"].astype(float)
            cm_normalized = cm / cm.sum(axis=1, keepdims=True) *100
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=d["classes"])
            disp.plot(cmap=plt.cm.Blues, ax=axes[i], colorbar=False)
            disp.ax_.set_xticklabels(d['classes'], rotation=45, ha='right', rotation_mode='anchor')
            axes[i].set_title(f"{d['model_name']} {d['confusion_matrix'].sum()} samples - Accuracy: {d['accuracy']['test']:.2f}")
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        if savename:
            plt.savefig(f"CONFUSION_MATRICES_COMPARISON_{savename}.pdf", bbox_inches='tight')
        if show:
            plt.savefig(f"temp.pdf", bbox_inches='tight')
            subprocess.Popen(["temp.pdf"],shell=True)
            time.sleep(1)
            os.remove(f"temp.pdf")
        plt.close()

        # Display Parameters in a table pdf with model name, accuracy, epochs, and params in a table without matplotlib (using only reportlab)
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
        from reportlab.lib import colors
        table_data = [["Model Name", "Accuracy", "Epochs", "Parameters"]]
        for d in data:
            params_str = "\n".join([f"{key}: {value}" for key, value in d["params"].items()]) if d["params"] is not None else "N/A"
            table_data.append([d["model_name"], f"{d['accuracy']['test']:.2f}", d["epochs"], params_str])
        pdf = SimpleDocTemplate(f"MODEL_COMPARISON_{savename}.pdf", pagesize=letter)
        table = Table(table_data, colWidths=[150, 100, 100, 250])
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ])
        table.setStyle(style)
        elements = [table]
        if show:
            pdf.build(elements)
            subprocess.Popen([f"MODEL_COMPARISON_{savename}.pdf"],shell=True)
            time.sleep(1)
            os.remove(f"MODEL_COMPARISON_{savename}.pdf")
        if savename:
            pdf.build(elements)
        # Open the generated PDF file

class TorchMLP(nn.Module):
    def __init__(self, hidden_layers_sizes=[300, 300, 200, 100, 50], activation=nn.ReLU, IO = (400, 5), num_epochs=500, batch_size=None,
                 plot_loss=False, loss_filename=None, verbose=True, x_val=None, y_val=None, learning_rate=1e-6, dropout_rate=0,
                 class_weights=[1.0, 1.0, 1.0, 1.0, 1.0]):

        # Appel du constructeur de la classe parente (nn.Module)
        super(TorchMLP, self).__init__()

        # On stocke le nombre de classes pour la confusion matrix
        self.epochs = num_epochs
        self.batch_size = batch_size
        self.plot_loss = plot_loss
        self.loss_filename = loss_filename
        self.verbose = verbose
        self.x_val = x_val
        self.y_val = y_val
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.num_classes = IO[1]
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

        # Create a sequential model
        self.model = nn.Sequential()

        # Add the input layer and the first hidden layer, along with the activation function and dropout
        self.model.add_module("input", nn.Linear(IO[0], hidden_layers_sizes[0]))
        self.model.add_module(f"act_input", activation())
        self.model.add_module(f"dropout_input", nn.Dropout(p=self.dropout_rate))

        # Create the hidden layers and add them to the model
        for layer in range(len(hidden_layers_sizes)-1):
            self.model.add_module(f"dense{layer+1}", nn.Linear(hidden_layers_sizes[layer], hidden_layers_sizes[layer+1]))
            self.model.add_module(f"act{layer+1}", activation())
            self.model.add_module(f"dropout{layer+1}", nn.Dropout(p=self.dropout_rate))
        
        # Add the output layer (without activation, since we'll use CrossEntropyLoss which applies softmax internally)
        self.model.add_module("output", nn.Linear(hidden_layers_sizes[-1], IO[1]))

    def forward(self, x):
        """
        This method defines the forward pass of the model. It takes an input tensor x and returns the output of the model (logits).
        Note: The softmax activation is not applied here because we will use CrossEntropyLoss which expects raw logits.
        Inputs - x: input tensor of shape (batch_size, input_features)
        Outputs - logits: output tensor of shape (batch_size, num_classes)
        """
        return self.model(x)

    def fit(self, x_train, y_train):
        """
        This method trains the model using the provided training data (x_train, y_train) and optionally evaluates on
        validation data (x_val, y_val) at each epoch. It also supports mini-batch training and can plot the loss curves after training.
        Inputs:
            - x_train: training features (numpy array or torch tensor)
            - y_train: training labels (numpy array or torch tensor)
        """

        self.train()

        # Convert inputs to torch tensors if they are numpy arrays
        if not isinstance(x_train, torch.Tensor):
            x_train = torch.tensor(x_train, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.long)

        # Check if validation data is provided and convert to torch tensors if necessary
        has_val = self.x_val is not None and self.y_val is not None
        if has_val:
            if not isinstance(self.x_val, torch.Tensor):
                self.x_val = torch.tensor(self.x_val, dtype=torch.float32)
            if not isinstance(self.y_val, torch.Tensor):
                self.y_val = torch.tensor(self.y_val, dtype=torch.long)

        # Define the loss function and optimizer

        loss_fn = nn.CrossEntropyLoss(self.class_weights)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # Determine effective batch size (full batch if None)
        n_samples = x_train.shape[0]
        effective_batch_size = n_samples if self.batch_size is None else self.batch_size

        # Lists to store loss values for plotting
        train_losses = []
        val_losses   = []

        # Training loop
        for n in range(self.epochs):

            # --- Shuffle data at each epoch ---
            perm = torch.randperm(n_samples)
            x_train = x_train[perm]
            y_train = y_train[perm]

            # --- Mini-batch loop ---
            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, n_samples, effective_batch_size):
                x_batch = x_train[start : start + effective_batch_size]
                y_batch = y_train[start : start + effective_batch_size]

                # Training step
                self.train()
                y_pred = self(x_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  # Gradient clipping to prevent exploding gradients
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            # Average loss over all batches for this epoch
            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)

            # --- Validation step ---
            if has_val:
                self.eval()
                with torch.no_grad():
                    val_loss = loss_fn(self(self.x_val), self.y_val).item()
                val_losses.append(val_loss)

            if self.verbose and n % 50 == 0:
                msg = f"Epoch {n}, Train Loss: {avg_train_loss:.4f}"
                if has_val:
                    msg += f", Val Loss: {val_loss:.4f}"
                print(msg)

        if self.plot_loss or self.loss_filename is not None:
            plt.figure(figsize=(8, 5))
            plt.plot(train_losses, label="Train Loss", color="steelblue")
            if has_val:
                plt.plot(val_losses, label="Val Loss", color="tomato")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss Curves")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            if self.loss_filename is not None:
                plt.savefig(self.loss_filename, bbox_inches='tight')
            if self.plot_loss:
                plt.show()
            plt.close()
        
        return train_losses, val_losses
    
    def predict_proba(self, x):
        """
        This method applies the softmax function to the output of the forward pass to get probabilities for each class.
        Inputs - x: input tensor of shape (batch_size, input_features)
        Outputs - probs: output tensor of shape (batch_size, num_classes) where each row sums to
                  1 and represents the predicted probabilities for each class.
        """
        return torch.softmax(self(x), dim=1)

    def score(self, x, y):
        self.eval()  # Désactive dropout etc.
        with torch.no_grad():  # Pas de gradient nécessaire
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.long)
            
            preds = self(x).argmax(dim=1)  # Classe prédite = argmax des logits
            accuracy = (preds == y).float().mean().item()
        return accuracy

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            y_pred = self(x).argmax(dim=1).numpy()
        return y_pred

    def get_confusion_matrix(self, x, y):
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.long)

            y_pred = self(x).argmax(dim=1).numpy()
            y_true = y.numpy()

        cm = confusion_matrix(y_true, y_pred)
        return cm

    def save_confusion_matrix(self, x_test, y_test, class_names=None, filename="CONFUSION_MATRIX.pdf", show=True):
        self.eval()
        with torch.no_grad():
            if not isinstance(x_test, torch.Tensor):
                x_test = torch.tensor(x_test, dtype=torch.float32)
            if not isinstance(y_test, torch.Tensor):
                y_test = torch.tensor(y_test, dtype=torch.long)

            y_pred = self(x_test).argmax(dim=1).numpy()
            y_true = y_test.numpy()

        if class_names is None:
            class_names = [str(i) for i in range(self.num_classes)]

        disp = ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            display_labels=class_names,
            cmap=plt.cm.Blues,
            normalize=None,
        )

        ticks = range(len(class_names))
        disp.ax_.set_xticks(ticks)
        disp.ax_.set_xticklabels(class_names, rotation=45, ha='right', rotation_mode='anchor')
        disp.ax_.set_yticks(ticks)
        disp.ax_.set_yticklabels(class_names)

        disp.ax_.set_xlabel('Predicted labels', fontsize=14, fontweight='bold')
        disp.ax_.set_ylabel('True labels', fontsize=14, fontweight='bold')

        plt.tight_layout()
        if filename:
            plt.savefig(filename, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def save_proba_class_x(self, x, y_true, class_idx, class_name=None, filename="PROBA_CLASS_X.pdf", show=True):
        """
        Plot the distribution of predicted probabilities for a specific class, split into TP, FN, FP.
        Inputs:
            - x         : input data (numpy array or torch tensor)
            - y_true    : true labels (numpy array or torch tensor)
            - class_idx : index of the class to analyze (e.g. 4 for gunshot)
            - class_name: display name for the class (e.g. "gunshot")
            - filename  : output filename
            - show      : whether to display the plot
        """
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if not isinstance(y_true, torch.Tensor):
                y_true = torch.tensor(y_true, dtype=torch.long)

            probs   = self.predict_proba(x).numpy()        # (n_samples, n_classes)
            y_pred  = probs.argmax(axis=1)                 # predicted class
            y_true  = y_true.numpy()

        label = class_name if class_name is not None else f"Class {class_idx}"

        # Masks
        is_true_positive  = (y_true == class_idx) & (y_pred == class_idx)   # vrai gunshot, prédit gunshot
        is_false_negative = (y_true == class_idx) & (y_pred != class_idx)   # vrai gunshot, prédit autre
        is_false_positive = (y_true != class_idx) & (y_pred == class_idx)   # pas gunshot, prédit gunshot

        probs_tp = probs[is_true_positive,  class_idx]
        probs_fn = probs[is_false_negative, class_idx]
        probs_fp = probs[is_false_positive, class_idx]

        probas_to_plot = [probs_tp, probs_fn, probs_fp]
        labels = [f"TP (n={len(probs_tp)})", f"FN (n={len(probs_fn)})", f"FP (n={len(probs_fp)})"]

        bins = np.linspace(0, 1, 25)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(probas_to_plot, bins=bins, alpha=0.7, color=["steelblue", "tomato", "orange"], edgecolor="black", label=labels)

        ax.set_xlabel(f"Predicted Probability — {label}", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"Probability Distribution for '{label}' — TP / FN / FP", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.4)
        plt.tight_layout()

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        if show:
            plt.savefig(f"temp.pdf", bbox_inches='tight')
            subprocess.Popen(["temp.pdf"],shell=True)
            time.sleep(1)
            os.remove(f"temp.pdf")
        plt.close()

    def GridSearch(self, x_train, y_train, param_grid, cv=5, verbose=True, savename="nounours_love_honing"):
        """
        This method performs a grid search over the specified hyperparameters in param_grid using cross-validation.
        Inputs:
            - x_train: training features (numpy array or torch tensor)
            - y_train: training labels (numpy array or torch tensor)
            - param_grid: dictionary where keys are hyperparameter names and values are lists of possible values
            - cv: number of cross-validation folds
            - verbose: whether to print progress
        Outputs:
            - best_params: dictionary of the best hyperparameters found
            - results: list of dicts with all combinations and their scores, sorted by mean_score
        """

        # Convert inputs to numpy for easier slicing during CV splits
        if isinstance(x_train, torch.Tensor):
            x_train = x_train.numpy()
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.numpy()

        n_samples = x_train.shape[0]

        # Placeholder for best parameters and best score
        best_params = None
        best_score  = -1.0
        results     = []

        # Generate all combinations of hyperparameters
        from itertools import product
        keeped_frequencies = param_grid.pop("keeped_frequencies", [(0,19)])
        background_range_list = param_grid.pop("add_background", [(-30,-27)])
        exposant_list = param_grid.pop("exposant", [0.5])
        keys, values = zip(*param_grid.items())
        all_combinations = [dict(zip(keys, combo)) for combo in product(*values)]
        n_combinations = len(all_combinations)*len(background_range_list)*len(keeped_frequencies)*len(exposant_list)

        # Precompute CV fold indices (stratified by shuffling)
        indices = np.random.permutation(n_samples)
        folds = np.array_split(indices, cv)  # cv roughly equal folds

        if verbose:
            print(f"Grid search over {n_combinations} combinations × {cv} folds "
                f"= {n_combinations * cv} trainings\n")
            
        idx_advance = 1
        for m in range(len(background_range_list)):
            for n in range(len(keeped_frequencies)):
                for o in range(len(exposant_list)):
                    x_train_augmented, y_train_augmented = add_background(x_train, y_train, attenuation_dB_range=background_range_list[m])
                    x_train_augmented = keep_frequencies(x_train_augmented, n_melvecs_to_keep=keeped_frequencies[n])**exposant_list[o]

                    for i, params in enumerate(all_combinations):

                        fold_scores = []
                        fold_cm = []
                        duration = time.time()

                        for fold_idx in range(cv):
                            
                            print("Current advance:", idx_advance, "/", n_combinations, "fold:", fold_idx+1, "/", cv)
                            # --- Build train/val split for this fold ---
                            val_indices   = folds[fold_idx]
                            train_indices = np.concatenate([folds[j] for j in range(cv) if j != fold_idx])

                            x_fold_train = x_train_augmented[train_indices]
                            y_fold_train = y_train_augmented[train_indices]
                            x_fold_val   = x_train_augmented[val_indices]
                            y_fold_val   = y_train_augmented[val_indices]

                            # --- Instantiate a fresh model with current params ---
                            # Separate IO from the rest since it's a constructor param but not a hyperparameter
                            # that changes between folds
                            IO = (x_fold_train.shape[1], len(np.unique(y_fold_train)))
                            model = TorchMLP(IO=IO, **params)
                            model.fit(x_fold_train, y_fold_train)

                            fold_score = model.score(x_fold_val, y_fold_val)
                            fold_scores.append(fold_score)

                            confusion_matrix = model.get_confusion_matrix(x_fold_val, y_fold_val)
                            fold_cm.append(confusion_matrix)

                        idx_advance += 1
                        duration = time.time() - duration
                        print("Duration for this combination:", duration, "seconds")
                        mean_score = np.mean(fold_scores)
                        std_score  = np.std(fold_scores)

                        results.append({
                            "params":     params,
                            "mean_score": mean_score,
                            "std_score":  std_score,
                            "fold_scores": fold_scores,
                            "fold_cm": fold_cm,
                            "frequencies": keeped_frequencies[n],
                            "background_range": background_range_list[m],
                            "exposant": exposant_list[o]
                        })

                        if verbose:
                            print(f"[{i+1}/{n_combinations}] {params}")
                            print(f"    → scores: {[f'{s:.4f}' for s in fold_scores]}")
                            print(f"    → mean: {mean_score:.4f} ± {std_score:.4f}\n")

                        if mean_score > best_score:
                            best_score  = mean_score
                            best_params = params

            # Sort results by mean_score descending for easy inspection
            results.sort(key=lambda r: r["mean_score"], reverse=True)

        if verbose:
            print("=" * 50)
            print(f"Best params : {best_params}")
            print(f"Best CV score : {best_score:.4f}")
        temp_var = time.ctime().replace(":", "_")
        pickle.dump(results, open(f"{savename}_GS_{temp_var}.pkl", "wb"))
        return best_params, results