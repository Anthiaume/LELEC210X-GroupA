from itertools import count

from student_fct import load_data, load_compacted_data, save_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import numpy as np
import pickle
import matplotlib.pyplot as plt

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

MLP_ephesos = False
if MLP_ephesos:
    print("Model MLP_ephesos not implemented yet")
# with open("hyperparameter_tuning dorystolon.pkl", "rb") as f:
#     hyperparameter_tuning = pickle.load(f)

# # Save all the mean_fit_time, param_activation, param_hidden_layer_sizes, param_learning_rate, param_max_iter ranked according to the mean_test_score in a md table
# results = hyperparameter_tuning.cv_results_
# sorted_indices = np.argsort(results['mean_test_score'])[::-1]  # Sort by mean_test_score in descending order

# with open("hyperparameter_results_dorystolon.md", "w") as f:
#     f.write("| mean_fit_time | param_activation | param_hidden_layer_sizes | param_learning_rate | param_max_iter | param_alpha | mean_test_score |\n")
#     f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
#     for i in sorted_indices:
#         f.write(f"| {results['mean_fit_time'][i]} | {results['param_activation'][i]} | {results['param_hidden_layer_sizes'][i]} | {results['param_learning_rate'][i]} | {results['param_max_iter'][i]} | {results['param_alpha'][i]} | {results['mean_test_score'][i]} |\n")# with open("hyperparameter_tuning bithynion 1.pkl", "rb") as f:

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

records = [("mcu13", "fisher", "local speakers - spec_20_20"),         # 5 classes, 122 samples per class
            ("mcu13", "vinikot", "JBL Flip 5 - Auguste - spec_20_20"),
            ("mcu13", "sud5", "local speakers - spec_20_20"),       # 5 classes, 122 samples per class
            ("mcu13", "sud11", "local speakers - spec_20_20 - Jonathan"),
            ("mcu13", "sud11", "local speakers - spec_20_20 - Raphael")]         # 5 classes, 122 samples per class]

data, labels = load_data(records)
data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)
x_train, x_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=0.3, random_state=42, shuffle=True, stratify=labels)



# --- Paramètres du meilleur modèle trouvé par GridSearch ---
best_params = {
    'hidden_layer_sizes': (300, 300, 200, 100, 50),
    'activation': 'relu',
    'alpha': 1e-4,
    'batch_size': 'auto',
    'learning_rate': 'constant',
    'random_state': 42,
}

N_EPOCHS = 500  # Nombre total d'itérations à tracer



# --- Initialisation du MLP avec warm_start ---
mlp = MLPClassifier(
    **best_params,
    max_iter=1,          # 1 epoch à la fois
    warm_start=True,     # conserve les poids entre les appels à fit()
    n_iter_no_change=N_EPOCHS,  # désactive l'early stopping interne
    tol=0.0,
)


train_accs = []
val_accs   = []

for epoch in range(N_EPOCHS):
    mlp.fit(x_train, y_train)
    
    # Accuracy sur le train
    train_accs.append(mlp.score(x_train, y_train))
    
    # Accuracy sur le test (validation)
    val_accs.append(mlp.score(x_test, y_test))
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{N_EPOCHS} — train acc: {train_accs[-1]:.3f} | val acc: {val_accs[-1]:.3f}")

train_losses = []
val_losses   = []

# --- Initialisation du MLP avec warm_start ---
mlp = MLPClassifier(
    **best_params,
    max_iter=1,          # 1 epoch à la fois
    warm_start=True,     # conserve les poids entre les appels à fit()
    n_iter_no_change=N_EPOCHS,  # désactive l'early stopping interne
    tol=0.0,
)


for epoch in range(N_EPOCHS):
    mlp.fit(x_train, y_train)
    
    # Training loss (log-loss, déjà calculé en interne)
    train_losses.append(mlp.loss_)
    
    # Validation loss : log-loss manuel sur x_test
    y_prob = mlp.predict_proba(x_test)
    # Clip pour éviter log(0)
    y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
    # One-hot encode y_test
    classes = mlp.classes_
    y_test_oh = np.zeros_like(y_prob)
    for i, label in enumerate(y_test):
        y_test_oh[i, np.where(classes == label)[0][0]] = 1
    val_loss = -np.mean(np.sum(y_test_oh * np.log(y_prob), axis=1))
    val_losses.append(val_loss)
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{N_EPOCHS} — train loss: {train_losses[-1]:.4f} | val loss: {val_losses[-1]:.4f}")












fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot([a * 100 for a in train_accs], color='steelblue', linewidth=2, label='Train')
ax1.plot([a * 100 for a in val_accs],   color='coral',    linewidth=2, linestyle='--', label='Validation')
ax1.set_title('Accuracy')
ax1.set_ylabel('%')
ax1.set_xlabel('Epoch')
ax1.set_ylim(0, 105)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(train_losses, color='steelblue', linewidth=2, label='Train')
ax2.plot(val_losses,   color='coral',    linewidth=2, linestyle='--', label='Validation')
ax2.set_title('Loss (log-loss)')
ax2.set_xlabel('Epoch')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('MLP Dorystolon — Training curves', fontsize=13)
plt.tight_layout()
plt.savefig('training_curves_dorystolon.png', dpi=150)
plt.show()