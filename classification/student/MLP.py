from itertools import count

from student_fct import load_data, load_compacted_data, save_confusion_matrix, add_background, suppress_low_frequencies, TorchMLP, ModelResults
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time

# Names of the models: amorium, bithynion, chios, dorystolon, ephesos, flaviopolis, gangra, halikarnassos, iconium, karthago, lebessos, mesembria, nicosia, ophis, philadelphia, quiza, rhodos, samos, tarsos

MLP_amorium = False # Clôturé le jeudi 26 février 2026 à 17h13
if MLP_amorium:
    """
    Model description:
    Basic MLP model with 5 classes
    Dataset: mcu13, vinikot, JBL Flip 5 - Auguste - spec_20_20 --> chainsaw, crackling fire, fireworks, gunshot
             mcu13, fisher, local speakers - spec_20_20 --> background
    Pas op !!! background est pas enregistré dans le même local que les autres classes
    Modèle clôturé le jeudi 26 février 2026 à 17h13
    """
    GS_amorium  = False
    Gen_amorium = False
    KF_amorium  = False

    records = [("mcu13", "vinikot", "JBL Flip 5 - Auguste - spec_20_20"), # chainsaw, crackling fire, fireworks, gunshot
               ("mcu13", "fisher", "local speakers - spec_20_20")       ] # background
    data, labels = load_data(records)
    print(labels)

    data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)
    x_train, x_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=0.3, random_state=42, shuffle=True, stratify=labels)
    
    if GS_amorium:
        parameters = {
            'hidden_layer_sizes': [(400, 400, 300, 200, 100, 50, 10), (300, 300, 200, 100, 50, 10), (200, 200, 100, 50, 10), (100, 100, 50, 10), (50, 50, 10)],
            'max_iter': [500], # Pas d'influence sensible sur les résultats
            'random_state': [42],
            "activation": ["relu"], # relu est clairement le meilleur
            "learning_rate": ["constant"] # pas de différence entre constant et adaptive, on laisse le paramètre par défaut
        }

        hyperparameter_tuning = GridSearchCV(MLPClassifier(), parameters, cv=5, n_jobs=-1)
        hyperparameter_tuning.fit(x_train, y_train)
        pickle.dump(hyperparameter_tuning, open("hyperparameter_tuning amorium 1.pkl", "wb"))

    if KF_amorium:
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        #mlp = MLPClassifier(hidden_layer_sizes=(200, 100, 50, 10), max_iter=500, random_state=42, activation="relu", learning_rate="constant")
        mlp = MLPClassifier(hidden_layer_sizes=(200, 200, 100, 50, 10), max_iter=500, random_state=42, activation="relu", learning_rate="constant")

        mean = []
        for train_index, val_index in kfold.split(x_train):
            x_train_fold, x_test_fold = x_train[train_index], x_train[val_index]
            y_train_fold, y_test_fold = y_train[train_index], y_train[val_index]

            # count labels in y_train_fold
            unique, counts = np.unique(y_train_fold, return_counts=True)
            print(f"Labels distribution in training fold: {dict(zip(unique, counts))}")

            
            mlp.fit(x_train_fold, y_train_fold)
            y_pred = mlp.predict(x_test_fold)
            score = accuracy_score(y_test_fold, y_pred)
            print(f"Fold score: {score}")
            mean.append(score)
        print(f"Mean score: {np.mean(mean)}")
    
    if Gen_amorium:
        mlp = MLPClassifier(hidden_layer_sizes=(200, 200, 100, 50, 10), max_iter=500, random_state=42, activation="relu", learning_rate="constant")
        mlp.fit(x_train, y_train)
        y_pred = mlp.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        print(f"Test score: {score}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        pickle.dump(mlp, open("MLP_amorium.pkl", "wb"))
        print("Model saved as MLP_amorium.pkl")
        print("classes: ", mlp.classes_)
        probas = mlp.predict_proba(x_test)
        probas = np.round(probas, decimals=2)
        # for i in range(len(y_test)):
        #     print(f"True label: {y_test[i]}, Predicted label: {y_pred[i]}, Predicted probabilities: chainsaw: {probas[i][0]}, fire: {probas[i][1]}, fireworks: {probas[i][2]}, gunshot: {probas[i][3]}")

MLP_bithynion = False # Clôturé le lundi 9 mars 2026 à 21h40
if MLP_bithynion:
    """
    Model description:
    MLP model with 5 classes
    Dataset: mcu13, vinikot, JBL Flip 5 - Auguste - spec_20_20 --> chainsaw, crackling fire, fireworks, gunshot, background
             mcu13, fisher, local speakers - spec_20_20 --> chainsaw, crackling fire, fireworks, gunshot, background
    Dataset équilibré
        - Objectif  : Etudier l'effet d'une PCA sur les performances du MLP et optimiser les hyperparamètres du MLP par GridSearchCV
        - Résultats : si les dernières hidden layers sont trop petites (~10), le PCA augmente les performances
                      si l'on corrige en augmentant la taille des dernières hidden layers, le PCA n'a plus d'influence sensible sur les performances
        - Conclusion : le PCA n'est pas nécessaire pour ce dataset, il n'améliore pas les performances du MLP de manière significative, et il est plus simple de ne pas l'utiliser
                       On garde donc un modèle MLP basique sans PCA, avec une architecture optimisée par GridSearchCV
                       Meilleure architecture: (300, 300, 200, 100, 50)
    Modèle clôturé le lundi 9 mars 2026 à 21h40
    """
    GS_bithynion  = False
    PCA_bithynion = False
    Gen_bithynion = True

    records = [("mcu13", "fisher", "local speakers - spec_20_20"),         # 5 classes, 122 samples per class
               ("mcu13", "vinikot", "JBL Flip 5 - Auguste - spec_20_20")]   # 5 classes, 111 samples per class
    data, labels = load_data(records)
    print(labels)

    data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)
    x_train, x_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=0.3, random_state=42, shuffle=True, stratify=labels)

    if PCA_bithynion:
        # n_pca = [350, 200, 150, 100, 50, 25, 10, 5]
        n_pca = [40, 35, 30, 25, 20, 15]
        hidden_layer_sizes = [(200, 200, 100, 50, 10), (400, 300, 200, 100, 50, 10), (50, 50, 50, 25, 25)]
        scores = []
        params = []
        for i in range(len(n_pca)):
            pca = PCA(n_components=n_pca[i])
            x_train_pca = pca.fit_transform(x_train)
            x_test_pca = pca.transform(x_test)
            for j in range(len(hidden_layer_sizes)):
                print("Advance : ", i * len(hidden_layer_sizes) + j, " / ", len(n_pca) * len(hidden_layer_sizes))
                model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes[j], max_iter=500, random_state=42, activation="relu", learning_rate="constant")
                model.fit(x_train_pca, y_train)
                prediction = model.predict(x_test_pca)
                score = accuracy_score(y_test, prediction)
                scores.append(score)
                params.append((n_pca[i], hidden_layer_sizes[j]))
        pickle.dump((scores, params), open("PCA_bithynion_scores_vdd.pkl", "wb"))

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(scores)), scores, marker='o')
        plt.title('MLP Accuracy with PCA on Bithynion Dataset')
        plt.xlabel('Configuration Index')
        plt.ylabel('Accuracy')
        plt.xticks(range(len(scores)), [f"PCA: {p[0]}, Hidden Layers: {p[1]}" for p in params], rotation=90)
        plt.grid()
        plt.tight_layout()
        plt.savefig("PCA_bithynion_scores_vdd.pdf")
        plt.show()

    if GS_bithynion:

        pca = PCA(n_components=35)
        x_train_pca = pca.fit_transform(x_train)
        x_test_pca = pca.transform(x_test)

        parameters = {
            'hidden_layer_sizes': [(300, 300, 200, 100, 50, 10), (300, 300, 200, 100, 50), (300, 300, 200, 100, 50, 25), (300, 300, 200, 100, 100), (300, 300, 200, 100, 100, 50, 50), (200, 200, 100, 50, 10), (100, 100, 100, 100, 50, 50, 10)], #(400, 400, 300, 200, 100, 50, 10), (50, 50, 50, 25, 25, 10), (35, 35, 20, 20, 10), (300, 300, 200, 100, 50, 10, 5), (300, 300, 200, 100, 50, 10, 1),  (100, 100, 50, 10), 
            'max_iter': [500], # Pas d'influence sensible sur les résultats
            'random_state': [42],
            "activation": ["relu"], # relu est clairement le meilleur
            "learning_rate": ["constant"] # pas de différence entre constant et adaptive, on laisse le paramètre par défaut
        }

        hyperparameter_tuning = GridSearchCV(MLPClassifier(), parameters, cv=5, n_jobs=-1)
        hyperparameter_tuning.fit(x_train, y_train)
        pickle.dump(hyperparameter_tuning, open("hyperparameter_tuning bithynion 1.pkl", "wb"))

    if Gen_bithynion:
        # Print performances
        mlp = MLPClassifier(hidden_layer_sizes=(300, 300, 200, 100, 50)	, max_iter=500, random_state=42, activation="relu", learning_rate="constant")
        mlp.fit(x_train, y_train)
        y_pred = mlp.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        print(f"Test score: {score}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        save_confusion_matrix(mlp, x_test, y_test, filename="confusion_matrix_bithynion.pdf", show=True)

        # Save model on all the dataset
        mlp.fit(data_normalized, labels)
        pickle.dump(mlp, open("MLP_bithynion.pkl", "wb"))
        print("Model saved as MLP_bithynion.pkl")

MLP_chios = False # Clôturé le lundi 23 mars 2026 à 1h20
if MLP_chios:
    records = [("mcu13", "vinikot", "JBL Flip 5 - Auguste - spec_20_20"), # chainsaw, crackling fire, fireworks, gunshot
               ("mcu13", "fisher", "local speakers - spec_20_20"),
               ("mcu13", "sud5", "local speakers - spec_20_20"),
               ("mcu13", "sud11", "local speakers - spec_20_20 - Jonathan"),
               ("mcu13", "sud11", "local speakers - spec_20_20 - Raphael")]         # 5 classes, 122 samples per class]
    compacting_levels = [1, 2, 3, 4, 5]
    score_testing     = []
    for j in range(1, 6):
        data, labels = load_compacted_data(records, n_samples_per_new_sample=j)
        print(f"Shapes {j} level of compacting:", data.shape, labels.shape)
        data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)
        x_train, x_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=0.3, random_state=42, shuffle=True, stratify=labels)

        # Validation set
        mlp = MLPClassifier(hidden_layer_sizes=(300, 300, 200, 100, 50)	, max_iter=500, random_state=42, activation="relu", learning_rate="constant")
        mlp.fit(x_train, y_train)
        y_pred = mlp.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        print(f"Test score with compacted data {j} samples per sample: {score}")
        score_testing.append(score)

        pickle.dump(mlp, open(f"MLP_chios_compacted_{j}.pkl", "wb"))
        # print("Confusion Matrix:")
        #print(confusion_matrix(y_test, y_pred))
        #save_confusion_matrix(mlp, x_test, y_test, filename="confusion_matrix_chios.pdf", show=True)
    label_size = 19
    ticks_size = 15
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(compacting_levels, score_testing, color = 'blue', lw=2, marker='o', markersize=8)
    ax.grid()
    ax.set_xlabel('Number of compacted melspectrograms', fontsize=label_size)
    ax.set_ylabel('Accuracy', fontsize=label_size)
    #ax.set_xlim(0, 15)
    ax.set_xticks(compacting_levels)
    ax.tick_params(axis='both', which='major', labelsize=ticks_size)
    plt.tight_layout()
    plt.savefig('MLP_chios_compacted.pdf')
    plt.show()

MLP_dorystolon = False # Clôturé le lundi 23 mars 2026 à 1h20
if MLP_dorystolon:
    Gen_dorystolon = False
    GS_dorystolon  = False

    records = [("mcu13", "fisher", "local speakers - spec_20_20"),         # 5 classes, 122 samples per class
               ("mcu13", "vinikot", "JBL Flip 5 - Auguste - spec_20_20"),
               ("mcu13", "sud5", "local speakers - spec_20_20"),       # 5 classes, 122 samples per class
               ("mcu13", "sud11", "local speakers - spec_20_20 - Jonathan"),
               ("mcu13", "sud11", "local speakers - spec_20_20 - Raphael")]         # 5 classes, 122 samples per class]

    data, labels = load_data(records)
    data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)
    x_train, x_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=0.3, random_state=42, shuffle=True, stratify=labels)

    if GS_dorystolon:
        print("Starting GridSearchCV for Dorystolon dataset...")
        parameters = {
            'hidden_layer_sizes': [(300, 300, 200, 100, 50), (300, 300, 200, 200, 100, 50), (300, 300, 200, 200, 100)],#(400, 400, 300, 200, 100, 50), (300, 300, 200, 100, 50), (200, 200, 100, 50), (300, 300, 200, 200, 100, 50), (300, 300, 200, 200, 100, 50)],
            'max_iter': [500], # Pas d'influence sensible sur les résultats
            'random_state': [42],
            "activation": ["relu"], # relu est clairement le meilleur
            "learning_rate": ["constant"], # pas de différence entre constant et adaptive, on laisse le paramètre par défaut
            "alpha": [0.0001, 0.001, 0.01],
            "batch_size": ["auto", len(labels)]
        }

        hyperparameter_tuning = GridSearchCV(MLPClassifier(), parameters, cv=5, n_jobs=-1)
        hyperparameter_tuning.fit(x_train, y_train)
        pickle.dump(hyperparameter_tuning, open("hyperparameter_tuning dorystolon.pkl", "wb"))


    if Gen_dorystolon:
        mlp = MLPClassifier(hidden_layer_sizes=(300, 300, 200, 100, 50)	, max_iter=500, random_state=42, activation="relu", learning_rate="constant")
        mlp.fit(x_train, y_train)
        y_pred = mlp.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        print(f"Test score: {score}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        save_confusion_matrix(mlp, x_test, y_test, filename="confusion_matrix_dorystolon.pdf", show=True)

MLP_ephesos = False # Clôturé le vendredi 10 avril 2026 à 12h43
if MLP_ephesos:
    records = [("mcu13", "fisher", "local speakers - spec_20_20"),         # 5 classes, 122 samples per class
               ("mcu13", "vinikot", "JBL Flip 5 - Auguste - spec_20_20"),
               ("mcu13", "sud5", "local speakers - spec_20_20"),       # 5 classes, 122 samples per class
               ("mcu13", "sud11", "local speakers - spec_20_20 - Jonathan"),
               ("mcu13", "sud11", "local speakers - spec_20_20 - Raphael")]         # 5 classes, 122 samples per class]

    GS_ephesos  = False
    Gen_ephesos = True

    # Load and preprocess data

    data_load_duration = time.time()
    data, labels = load_data(records)
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    MLP_ephesos_classes = le.classes_
    data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)
    data_normalized = data_normalized

    data_normalized, labels_encoded = add_background(data_normalized=data_normalized, labels=labels_encoded, attenuation_dB_range=(-20, -15))

    x_train, x_test, y_train, y_test = train_test_split(data_normalized, labels_encoded, test_size=0.3, random_state=42, shuffle=True, stratify=labels_encoded)
    # x_train, x_val, y_train, y_val   = train_test_split(x_train, y_train, test_size=0.3, random_state=32, shuffle=True, stratify=y_train)
    print("Data loaded and preprocessed for Ephesos dataset, duration: {:.2f} seconds.".format(time.time() - data_load_duration))


    if GS_ephesos:


        param_grid = {
            'hidden_layers_sizes': [(600, 300, 100)],#, [300, 300, 200, 100, 50], [600, 600, 300, 100]],
            "activation": [nn.ReLU], # relu est clairement le meilleur
            "IO": [(400, 5)],
            'num_epochs': [150],
            "batch_size": [len(labels_encoded)],
            "plot_loss": [False],
            "loss_filename": [None],
            "verbose": [False],
            "learning_rate": [1e-3],
            "dropout_rate": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        }

        GS_MLP_ephesos = TorchMLP()
        GS_MLP_ephesos.GridSearch(x_train, y_train, param_grid, cv=3, verbose=False)

    if Gen_ephesos:
        print(f"\n\n\n{"#"*90 + "\n"}Gen_ephesos:\n\nTraining MLP on Ephesos dataset with PyTorch implementation...")

        # Train the model and measure the training time
        duration = time.time()
        mlp = TorchMLP(hidden_layers_sizes=[600, 300, 100], activation=nn.ReLU, IO=(400, 5),  num_epochs=500, batch_size=len(labels_encoded),
                       learning_rate=1e-3, dropout_rate=0.5,
                       x_val=x_test, y_val=y_test, plot_loss=True, verbose=True,
                       loss_filename="LOSS_CURVES_ephesos_pytorch.pdf")
        mlp.fit(x_train, y_train)
        duration = time.time() - duration

        # Print performances
        train_acc = mlp.score(x_train, y_train)
        test_acc  = mlp.score(x_test, y_test)
        print(f"Accuracy train : {train_acc:.4f}")
        print(f"Accuracy test  : {test_acc:.4f}" )
        mlp.save_confusion_matrix(x_test, y_test, class_names=MLP_ephesos_classes, show=True, filename="confusion_matrix_ephesos_pytorch.pdf")
        
        # Save model on all the dataset
        mlp = TorchMLP(hidden_layers_sizes=[600, 300, 100], activation=nn.ReLU, IO=(400, 5),  num_epochs=150, batch_size=len(labels_encoded),
                       learning_rate=1e-3, dropout_rate=0.5, plot_loss=False, verbose=True,
                       loss_filename=None)
        mlp.fit(data_normalized, labels_encoded)
        pickle.dump(mlp, open("MLP_ephesos_pytorch.pkl", "wb"))

        print(f"Training time: {duration:.2f} seconds")
        print(f"Model trained and confusion matrix saved for Ephesos dataset with PyTorch implementation.{"\n" + "#"*90}\n\n\n")

MLP_flaviopolis = False # Clôturé le vendredi 10 avril à 12h43
if MLP_flaviopolis:
    records = [("mcu13", "fisher", "local speakers - spec_20_20"),         # 5 classes, 122 samples per class
               ("mcu13", "vinikot", "JBL Flip 5 - Auguste - spec_20_20"),
               ("mcu13", "sud5", "local speakers - spec_20_20"),       # 5 classes, 122 samples per class
               ("mcu13", "sud11", "local speakers - spec_20_20 - Jonathan"),
               ("mcu13", "sud11", "local speakers - spec_20_20 - Raphael")]         # 5 classes, 122 samples per class]

    Freq_step_flaviopolis = False
    Gen_flaviopolis = True
    Exp_step_flaviopolis = False

    ### Load and preprocess data

    # Load data and labels
    data_load_duration = time.time()
    data, labels = load_data(records)

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    MLP_flaviopolis_classes = le.classes_

    def load_data_flaviopolis(data=None, labels_encoded=None, n_melvecs_to_suppress=10, bool_add_background=False):
        # Normalize and suppress low frequencies in the data
        data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)
        data_normalized_HF = suppress_low_frequencies(data=data_normalized, n_melvecs_to_suppress=n_melvecs_to_suppress)


        # Suppress low frequencies in the data and then normalize the high frequencies part
        data_HF = suppress_low_frequencies(data=data, n_melvecs_to_suppress=n_melvecs_to_suppress)
        data_HF_normalized = data_HF / np.linalg.norm(data, axis=1, keepdims=True)

        # Add background in the training data with random attenuation between -20 and -15 dB
        if bool_add_background:
            data_normalized_HF, labels_e = add_background(data_normalized=data_normalized_HF, labels=labels_encoded, attenuation_dB_range=(-20, -15))
            data_HF_normalized, labels_e = add_background(data_normalized=data_HF_normalized, labels=labels_encoded, attenuation_dB_range=(-20, -15))
        else:
            labels_e = labels_encoded.copy()

        return data_normalized_HF, data_HF_normalized, labels_e

    print("Data loaded and preprocessed for Flaviopolis dataset, duration: {:.2f} seconds.".format(time.time() - data_load_duration))

    if Freq_step_flaviopolis:

        n_melvecs_to_suppress_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
        N_HF_train = []
        N_HF_test  = []
        HF_N_train = []
        HF_N_test  = []
        
        for n_melvecs_to_suppress in n_melvecs_to_suppress_list:
            data_normalized_HF, data_HF_normalized, labels_encoded = load_data_flaviopolis(n_melvecs_to_suppress=n_melvecs_to_suppress, bool_add_background=False)
            x_train_N_HF, x_test_N_HF, y_train_N_HF, y_test_N_HF = train_test_split(data_normalized_HF, labels_encoded, test_size=0.3, random_state=42, shuffle=True, stratify=labels_encoded)
            x_train_HF_N, x_test_HF_N, y_train_HF_N, y_test_HF_N = train_test_split(data_HF_normalized, labels_encoded, test_size=0.3, random_state=42, shuffle=True, stratify=labels_encoded)

            n_inputs = 400 - n_melvecs_to_suppress * 20
            
            # N_HF
            duration = time.time()
            mlp = TorchMLP(hidden_layers_sizes=[300, 300, 200, 100, 50], activation=nn.ReLU, IO=(n_inputs, 5),  num_epochs=150, batch_size=len(labels_encoded),
                        learning_rate=1e-3, dropout_rate=0.25,
                        x_val=x_test_N_HF, y_val=y_test_N_HF, plot_loss=False, verbose=True,
                        loss_filename="LOSS_CURVES_ephesos_pytorch.pdf")
            mlp.fit(x_train_N_HF, y_train_N_HF)
            duration = time.time() - duration
            train_acc = mlp.score(x_train_N_HF, y_train_N_HF)
            test_acc  = mlp.score(x_test_N_HF, y_test_N_HF)
            print(f"Accuracy train : {train_acc:.4f}")
            print(f"Accuracy test  : {test_acc:.4f}" )
            mlp.save_confusion_matrix(x_test_N_HF, y_test_N_HF, class_names=MLP_ephesos_classes, show=False, filename=f"confusion_matrix_ephesos_pytorch_N_HF_{n_melvecs_to_suppress}.pdf")
            N_HF_train.append(train_acc)
            N_HF_test.append(test_acc)
            print(f"Training time: {duration:.2f} seconds")

            # HF_N
            mlp = TorchMLP(hidden_layers_sizes=[300, 300, 200, 100, 50], activation=nn.ReLU, IO=(n_inputs, 5),  num_epochs=150, batch_size=len(labels_encoded),
                        learning_rate=1e-3, dropout_rate=0.25,
                        x_val=x_test_HF_N, y_val=y_test_HF_N, plot_loss=False, verbose=True,
                        loss_filename=None)
            mlp.fit(x_train_HF_N, y_train_HF_N)
            train_acc = mlp.score(x_train_HF_N, y_train_HF_N)
            test_acc  = mlp.score(x_test_HF_N, y_test_HF_N)
            print(f"Accuracy train : {train_acc:.4f}")
            print(f"Accuracy test  : {test_acc:.4f}" )
            mlp.save_confusion_matrix(x_test_HF_N, y_test_HF_N, class_names=MLP_ephesos_classes, show=False, filename=f"confusion_matrix_ephesos_pytorch_HF_N_{n_melvecs_to_suppress}.pdf")
            HF_N_train.append(train_acc)
            HF_N_test.append(test_acc)
            print(f"Training time: {duration:.2f} seconds")
        # Plot the results
        label_size = 19
        ticks_size = 15
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(n_melvecs_to_suppress_list, N_HF_train, color = 'blue', lw=2, marker='o', markersize=8, label="N_HF train")
        ax.plot(n_melvecs_to_suppress_list, N_HF_test, color = 'green', lw=2, marker='o', markersize=8, label="N_HF test")
        ax.plot(n_melvecs_to_suppress_list, HF_N_train, color = 'orange', lw=2, marker='o', markersize=8, label="HF_N train")
        ax.plot(n_melvecs_to_suppress_list, HF_N_test, color = 'red', lw=2, marker='o', markersize=8, label="HF_N test")
        ax.grid()
        ax.set_xlabel('Number of compacted melspectrograms', fontsize=label_size)
        ax.set_ylabel('Accuracy', fontsize=label_size)
        ax.set_xticks(n_melvecs_to_suppress_list)
        ax.tick_params(axis='both', which='major', labelsize=ticks_size)
        ax.legend(fontsize=label_size)
        plt.tight_layout()
        plt.savefig('MLP_flaviopolis_freq_step.pdf')
        plt.show()

        pickle.dump((n_melvecs_to_suppress_list, N_HF_train, N_HF_test, HF_N_train, HF_N_test), open("MLP_flaviopolis_freq_step_results.pkl", "wb"))

    if Exp_step_flaviopolis:


        exp_list = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.2, 2]
        exp_train = []
        exp_test  = []
        n_melvecs_to_suppress = 7

        for exposant in exp_list:
            print("Advance : ", 1+exp_list.index(exposant), " / ", len(exp_list))
            data_normalized_HF, data_HF_normalized, labels_encoded = load_data_flaviopolis(data=data, labels_encoded=labels_encoded, n_melvecs_to_suppress=n_melvecs_to_suppress, bool_add_background=False)
            
            if exposant == "log":
                data_normalized_HF = np.log(data_normalized_HF + 1e-8)
            elif exposant == "log10":
                data_normalized_HF = np.log10(data_normalized_HF + 1e-8)
            else:
                data_normalized_HF = data_normalized_HF ** exposant

            x_train_N_HF, x_test_N_HF, y_train_N_HF, y_test_N_HF = train_test_split(data_normalized_HF, labels_encoded, test_size=0.3, random_state=42, shuffle=True, stratify=labels_encoded)

            n_inputs = 400 - n_melvecs_to_suppress * 20
            
            # N_HF
            duration = time.time()
            mlp = TorchMLP(hidden_layers_sizes=[300, 300, 200, 100, 50], activation=nn.ReLU, IO=(n_inputs, 5),  num_epochs=150, batch_size=len(labels_encoded),
                        learning_rate=1e-3, dropout_rate=0.25,
                        x_val=x_test_N_HF, y_val=y_test_N_HF, plot_loss=False, verbose=True,
                        loss_filename=None)
            mlp.fit(x_train_N_HF, y_train_N_HF)
            duration = time.time() - duration
            train_acc = mlp.score(x_train_N_HF, y_train_N_HF)
            test_acc  = mlp.score(x_test_N_HF, y_test_N_HF)
            print(f"Accuracy train : {train_acc:.4f}")
            print(f"Accuracy test  : {test_acc:.4f}" )
            # mlp.save_confusion_matrix(x_test_N_HF, y_test_N_HF, class_names=MLP_flaviopolis_classes, show=False, filename=f"confusion_matrix_flaviopolis_pytorch_N_HF_{n_melvecs_to_suppress}.pdf")
            exp_train.append(train_acc)
            exp_test.append(test_acc)
            print(f"Training time: {duration:.2f} seconds")

        # Plot the results
        label_size = 19
        ticks_size = 15
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(exp_list, exp_train, color = 'blue', lw=2, marker='o', markersize=8, label="Exp train")
        ax.plot(exp_list, exp_test, color = 'green', lw=2, marker='o', markersize=8, label="Exp test")
        ax.grid()
        ax.set_xlabel("Exponent", fontsize=label_size)
        ax.set_ylabel('Accuracy', fontsize=label_size)
        # ax.set_xticks(exp_list.remove("log"))
        ax.tick_params(axis='both', which='major', labelsize=ticks_size)
        ax.legend(fontsize=label_size)
        plt.tight_layout()
        plt.savefig('MLP_flaviopolis_exp_step.pdf')
        plt.show()

        pickle.dump((exp_list, exp_train, exp_test), open("MLP_flaviopolis_exp_step_results.pkl", "wb"))

    if Gen_flaviopolis:

        n_melvecs_to_suppress_list = [0]#, 3, 5, 7, 9]
        
        exposant_list = [0.5]#, 0.5, 0.7, 0.9, 1, 1.5]

        for n_melvecs_to_suppress in n_melvecs_to_suppress_list:
            for exposant in exposant_list:
                inputs = 400 - n_melvecs_to_suppress * 20
                data_normalized_HF, data_HF_normalized, labels_encoded = load_data_flaviopolis(data=data, labels_encoded=labels_encoded, n_melvecs_to_suppress=n_melvecs_to_suppress, bool_add_background=False)

                data_normalized_HF = data_normalized_HF ** exposant
                # Split data into training and testing sets
                x_train_N_HF, x_test_N_HF, y_train_N_HF, y_test_N_HF = train_test_split(data_normalized_HF, labels_encoded, test_size=0.3, random_state=42, shuffle=True, stratify=labels_encoded)

                print(f"\n\n\n{"#"*90 + "\n"}Gen_flaviopolis:\n\nTraining MLP on Flaviopolis dataset with PyTorch implementation...")

                duration = time.time()
                mlp = TorchMLP(hidden_layers_sizes=[600, 300, 100], activation=nn.ReLU, IO=(inputs, 5),  num_epochs=500, batch_size=len(labels_encoded),
                            learning_rate=1e-3, dropout_rate=0.5,
                            x_val=x_test_N_HF, y_val=y_test_N_HF, plot_loss=True, verbose=True,
                            loss_filename="LOSS_CURVES_flaviopolis_pytorch.pdf")
                mlp.fit(x_train_N_HF, y_train_N_HF)
                duration = time.time() - duration
                print(f"Training time: {duration:.2f} seconds")

                train_acc = mlp.score(x_train_N_HF, y_train_N_HF)
                test_acc  = mlp.score(x_test_N_HF, y_test_N_HF)
                diff_acc = train_acc - test_acc

                print(f"Accuracy train : {train_acc:.4f}")
                print(f"Accuracy test  : {test_acc:.4f}" )
                print(f"Difference train - test accuracy: {diff_acc:.4f}")
                mlp.save_confusion_matrix(x_test_N_HF, y_test_N_HF, class_names=MLP_flaviopolis_classes, show=True, filename="confusion_matrix_flaviopolis_pytorch.pdf")

                # Save model on all the dataset
                mlp = TorchMLP(hidden_layers_sizes=[600, 300, 100], activation=nn.ReLU, IO=(inputs, 5),  num_epochs=150, batch_size=len(labels_encoded),
                    learning_rate=1e-3, dropout_rate=0.25, plot_loss=False, verbose=True, loss_filename=None)
                mlp.fit(data_normalized_HF, labels_encoded)
                pickle.dump(mlp, open("MLP_flaviopolis_pytorch.pkl", "wb"))

                print(f"Model trained and confusion matrix saved for Flaviopolis dataset with PyTorch implementation.{"\n" + "#"*90}\n\n\n")

MLP_gangra = False # Clôturé le vendredi 10 avril à 12h43
if MLP_gangra:
    records = [("mcu13", "fisher", "local speakers - spec_20_20"),         # 5 classes, 122 samples per class
               ("mcu13", "vinikot", "JBL Flip 5 - Auguste - spec_20_20"),
               ("mcu13", "sud5", "local speakers - spec_20_20"),       # 5 classes, 122 samples per class
               ("mcu13", "sud11", "local speakers - spec_20_20 - Jonathan"),
               ("mcu13", "sud11", "local speakers - spec_20_20 - Raphael")]         # 5 classes, 122 samples per class]

    Gen_gangra = True

    ### Load and preprocess data

    # Load data and labels
    data_load_duration = time.time()
    data, labels = load_data(records)

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    MLP_gangra_classes = le.classes_

    def load_data_gangra(n_melvecs_to_suppress=10, bool_add_background=False):
        # Normalize data
        data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)
        data_HF = suppress_low_frequencies(data=data, n_melvecs_to_suppress=n_melvecs_to_suppress)

        # Suppress low frequencies in the data
        data_normalized_HF = suppress_low_frequencies(data=data_normalized, n_melvecs_to_suppress=n_melvecs_to_suppress)
        data_HF_normalized = data_HF / np.linalg.norm(data, axis=1, keepdims=True)

        # Add background in the training data with random attenuation between -20 and -15 dB
        if bool_add_background:
            data_normalized_HF, labels_e = add_background(data_normalized=data_normalized_HF, labels=labels_encoded, attenuation_dB_range=(-20, -15))
            data_HF_normalized, labels_e = add_background(data_normalized=data_HF_normalized, labels=labels_encoded, attenuation_dB_range=(-20, -15))

        return data_normalized_HF, data_HF_normalized, labels_e

    print("Data loaded and preprocessed for Gangra dataset, duration: {:.2f} seconds.".format(time.time() - data_load_duration))

    if Gen_gangra:

        n_melvecs_to_suppress_list = [7]#, 3, 5, 7, 9]
        
        exposant_list = [0.5]#, 0.5, 0.7, 0.9, 1, 1.5]

        for n_melvecs_to_suppress in n_melvecs_to_suppress_list:
            for exposant in exposant_list:
                inputs = 400 - n_melvecs_to_suppress * 20
                data_normalized_HF, data_HF_normalized, labels_encoded = load_data_gangra(n_melvecs_to_suppress=n_melvecs_to_suppress, bool_add_background=True)

                data_normalized_HF = data_normalized_HF ** exposant
                # Split data into training and testing sets
                x_train_N_HF, x_test_N_HF, y_train_N_HF, y_test_N_HF = train_test_split(data_normalized_HF, labels_encoded, test_size=0.3, random_state=42, shuffle=True, stratify=labels_encoded)
                # x_train_HF_N, x_test_HF_N, y_train_HF_N, y_test_HF_N = train_test_split(data_HF_normalized, labels_encoded, test_size=0.3, random_state=42, shuffle=True, stratify=labels_encoded)

                print(f"\n\n\n{"#"*90 + "\n"}Gen_gangra:\n\nTraining MLP on Gangra dataset with PyTorch implementation...")

                duration = time.time()
                mlp = TorchMLP(hidden_layers_sizes=[600, 300, 100], activation=nn.ReLU, IO=(inputs, 5),  num_epochs=1000, batch_size=len(labels_encoded),
                            learning_rate=1e-3, dropout_rate=0.25,
                            x_val=x_test_N_HF, y_val=y_test_N_HF, plot_loss=True, verbose=True,
                            loss_filename="LOSS_CURVES_gangra_pytorch.pdf")
                mlp.fit(x_train_N_HF, y_train_N_HF)
                duration = time.time() - duration
                train_acc = mlp.score(x_train_N_HF, y_train_N_HF)
                test_acc  = mlp.score(x_test_N_HF, y_test_N_HF)
                diff_acc = train_acc - test_acc
                print(f"Accuracy train : {train_acc:.4f}")
                print(f"Accuracy test  : {test_acc:.4f}" )
                print(f"Difference train - test accuracy: {diff_acc:.4f}")
                mlp.save_confusion_matrix(x_test_N_HF, y_test_N_HF, class_names=MLP_gangra_classes, show=True, filename="confusion_matrix_gangra_pytorch.pdf")
                mlp = TorchMLP(hidden_layers_sizes=[600, 300, 100], activation=nn.ReLU, IO=(inputs, 5),  num_epochs=200, batch_size=len(labels_encoded),
                    learning_rate=1e-3, dropout_rate=0.5, plot_loss=False, verbose=True, loss_filename=None)
                mlp.fit(data_normalized_HF, labels_encoded)
                pickle.dump(mlp, open("MLP_gangra_pytorch.pkl", "wb"))

                print(f"Training time: {duration:.2f} seconds")
                print(f"Model trained and confusion matrix saved for Gangra dataset with PyTorch implementation.{"\n" + "#"*90}\n\n\n")

MLP_halikarnassos = True
if MLP_halikarnassos:
    """
    Objectifs :
    - Suppression de fréquences
    - Impact de la modification de l'exposant dans la normalisation (x^exposant) sur les performances du MLP
    - architecture du MLP : nombre de couches, nombre de neurones par couche, taux de dropout

    --> Objectif final: avoir de bonnes loss curves (perte d'entraînement et de validation qui diminuent de manière régulière et qui convergent vers une valeur basse), une bonne matrice de confusion (peu de confusions entre les classes) et un écart raisonnable entre la précision d'entraînement et de test (pas de surapprentissage).
    
    """

    records = [("mcu13", "fisher", "local speakers - spec_20_20"),         # 5 classes, 122 samples per class
               ("mcu13", "vinikot", "JBL Flip 5 - Auguste - spec_20_20"),
               ("mcu13", "sud5", "local speakers - spec_20_20"),       # 5 classes, 122 samples per class
               ("mcu13", "sud11", "local speakers - spec_20_20 - Jonathan"),
               ("mcu13", "sud11", "local speakers - spec_20_20 - Raphael")]         # 5 classes, 122 samples per class]

    Gen_halikarnassos = True

    ### Load and preprocess data

    # Load data and labels
    data, labels, labels_encoded, MLP_halikarnassos_classes = load_data(records)

    # Function to load and preprocess data for Halikarnassos dataset, with options for frequency suppression and background addition
    def load_data_halikarnassos(data, labels_encoded, n_melvecs_to_suppress=10, bool_add_background=False, attenuation_dB_range_list=((-20, -15))):
        # Normalize data
        data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)

        # Suppress low frequencies in the data
        data_normalized_HF = suppress_low_frequencies(data=data_normalized, n_melvecs_to_suppress=n_melvecs_to_suppress)

        # Add background in the training data with random attenuation between -20 and -15 dB
        if bool_add_background:

            for attenuation_dB_range in attenuation_dB_range_list:
                if attenuation_dB_range == attenuation_dB_range_list[0]:
                    data_normalized_HF_background, labels_encoded_background = add_background(data_normalized=data_normalized_HF, labels=labels_encoded,
                                                                      attenuation_dB_range=attenuation_dB_range)
                else:
                    new_data, new_labels = add_background(  data_normalized=data_normalized_HF, labels=labels_encoded,
                                                            attenuation_dB_range=attenuation_dB_range)
                    data_normalized_HF_background = np.concatenate((data_normalized_HF_background, new_data), axis=0)
                    labels_encoded_background = np.concatenate((labels_encoded_background, new_labels), axis=0) 
        return data_normalized_HF if not bool_add_background else data_normalized_HF_background, labels_encoded if not bool_add_background else labels_encoded_background



    if Gen_halikarnassos:

        print(f"\n\n\n{"#"*90 + "\n"}Gen_halikarnassos:\n\nTraining MLP on Halikarnassos dataset with PyTorch implementation...")

        # Params
        model_name = "Halikarnassos_v1"
        n_melvecs_to_suppress = 0
        exposant              = 0.5
        bool_add_background   = True
        n_epochs              = 1500
        hidden_layers_sizes = [600, 300, 100]
        activation = nn.ReLU
        bathch_size = len(labels_encoded)
        learning_rate = 1e-3
        dropout_rate = 0.5
        attenuation_dB_range_list = [(-20, -15), (-15, -10), (-10, -5), (-5, 0), (0, 5)]



        # Paramètres secondaires
        inputs = 400 - n_melvecs_to_suppress * 20
        IO = (inputs, 5)



        # Load and preprocess data with specified parameters
        data_normalized_HF, labels_encoded = load_data_halikarnassos(data, labels_encoded, n_melvecs_to_suppress=n_melvecs_to_suppress,
                                                                     bool_add_background=bool_add_background,
                                                                     attenuation_dB_range_list=attenuation_dB_range_list)
        data_normalized_HF = data_normalized_HF ** exposant
        x_train, x_test, y_train, y_test = train_test_split(data_normalized_HF, labels_encoded, test_size=0.3, random_state=42, shuffle=True, stratify=labels_encoded)



        # Training the model
        duration = time.time()
        mlp = TorchMLP(hidden_layers_sizes=hidden_layers_sizes, activation=activation, IO=IO,  num_epochs=n_epochs, batch_size=bathch_size,
                    learning_rate=learning_rate, dropout_rate=dropout_rate,
                    x_val=x_test, y_val=y_test, plot_loss=True, verbose=True,
                    loss_filename="LOSS_CURVES_halikarnassos_pytorch.pdf")
        train_losses, test_losses = mlp.fit(x_train, y_train)
        duration = time.time() - duration
        print(f"Training time: {duration:.2f} seconds")

        # Print performances and save confusion matrix
        train_acc = mlp.score(x_train, y_train)
        test_acc  = mlp.score(x_test, y_test)
        diff_acc = train_acc - test_acc

        print(f"Accuracy train : {train_acc:.4f}")
        print(f"Accuracy test  : {test_acc:.4f}" )
        print(f"Difference train - test accuracy: {diff_acc:.4f}")

        mlp.save_confusion_matrix(x_test, y_test, class_names=MLP_halikarnassos_classes, show=True, filename="confusion_matrix_halikarnassos_pytorch.pdf")
        
        # Save model results

        results = ModelResults(model_name = model_name,
                               model=mlp,
                               accuracy={"train": train_acc, "test": test_acc},
                               confusion_matrix=mlp.get_confusion_matrix(x_test, y_test),
                               classes=MLP_halikarnassos_classes,
                               epochs=n_epochs,
                               loss_curves={"train": train_losses, "test": test_losses},
                               params={
                                    "hidden_layers_sizes": hidden_layers_sizes,
                                    "learning_rate": learning_rate,
                                    "dropout_rate": dropout_rate,
                                    "attenuation_dB_range_list": attenuation_dB_range_list,
                                    "exposant": exposant,
                                    "n_melvecs_to_suppress": n_melvecs_to_suppress,
                                    "bool_add_background": bool_add_background,
                                    "batch_size": bathch_size,
                                    "activation": activation.__name__ ,
                                    "IO": IO,
                                    "n_melvecs_to_suppress": n_melvecs_to_suppress,
                                    "n_epochs": n_epochs
                               })
        results.save(f"results_{model_name}.pkl")
        # Save model on all the dataset
        # mlp = TorchMLP(hidden_layers_sizes=[600, 300, 100], activation=nn.ReLU, IO=(inputs, 5),  num_epochs=200, batch_size=len(labels_encoded),
        #     learning_rate=1e-3, dropout_rate=0.5, plot_loss=False, verbose=True, loss_filename=None)
        # mlp.fit(data_normalized_HF, labels_encoded)
        # pickle.dump(mlp, open("MLP_gangra_pytorch.pkl", "wb"))

        # print(f"Model trained and confusion matrix saved for Gangra dataset with PyTorch implementation.{"\n" + "#"*90}\n\n\n")
        # print(f"The model's classes are: {MLP_gangra_classes}")

# ModelResults().load_and_display(filenames=["results_Halikarnassos_v1.pkl"])

# with open("ephesos_GS_Mon Apr 13 01_53_50 2026.pkl", "rb") as f:
#     hyperparameter_tuning = pickle.load(f)
# # trier le dictionnaire hyperparameter_tuning selon "mean_score"
# # hyperparameter_tuning = sorted(hyperparameter_tuning, key=lambda x: hyperparameter_tuning["mean_score"], reverse=True)
# for element in hyperparameter_tuning:
#     print(f"Mean score: {element['mean_score']:.4f}, learning rate: {element['params']['learning_rate']}, dropout rate: {element['params']['dropout_rate']}, hidden layers sizes: {element['params']['hidden_layers_sizes']}")#, Params: {element['params']}")