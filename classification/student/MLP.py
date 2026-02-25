from load_data import load_data
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pickle

# Names of the models: amorium, bithynion, chios, dorystolon, ephesos, flaviopolis, gangra, halikarnassos, iconium, karthago, lebessos, mesembria, nicosia, ophis, philadelphia, quiza, rhodos, samos, tarsos

MLP_amorium = False
if MLP_amorium:
    """
    Model description:
    Basic MLP model with 5 classes
    Dataset: mcu13, vinikot, JBL Flip 5 - Auguste - spec_20_20
    """
    GS_amorium  = False
    Gen_amorium = False
    KF_amorium  = False

    mcu = "mcu13"
    locals = ["vinikot"]
    speakers = ["JBL Flip 5 - Auguste - spec_20_20"] 
    data, labels = load_data(mcu, locals, speakers)

    data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)
    x_train, x_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=0.3, random_state=42, shuffle=True)

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

MLP_bithynion = False
if MLP_bithynion:
    print("not implemented yet")

# with open("hyperparameter_tuning amorium 1.pkl", "rb") as f:
#     hyperparameter_tuning = pickle.load(f)

# # Save all the mean_fit_time, param_activation, param_hidden_layer_sizes, param_learning_rate, param_max_iter ranked according to the mean_test_score in a md table
# results = hyperparameter_tuning.cv_results_
# sorted_indices = np.argsort(results['mean_test_score'])[::-1]  # Sort by mean_test_score in descending order

# with open("hyperparameter_results.md", "w") as f:
#     f.write("| mean_fit_time | param_activation | param_hidden_layer_sizes | param_learning_rate | param_max_iter | mean_test_score |\n")
#     f.write("| --- | --- | --- | --- | --- | --- |\n")
#     for i in sorted_indices:
#         f.write(f"| {results['mean_fit_time'][i]} | {results['param_activation'][i]} | {results['param_hidden_layer_sizes'][i]} | {results['param_learning_rate'][i]} | {results['param_max_iter'][i]} | {results['mean_test_score'][i]} |\n")