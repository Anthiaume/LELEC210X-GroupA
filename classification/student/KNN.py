from student_fct import load_data, save_confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
#import principal component analysis (PCA) for dimensionality reduction if needed
from sklearn.decomposition import PCA
# model names: albaniana, blariacum, carvium, daventria, elenio, fectio, ganuenta, horgana, matilo, nigrum, praetorium, sablones, tablis, ulpia, venlonum

KNN_albaniana = False # Modèle clôturé le jeudi 12 mars 2026 à 00h48
if KNN_albaniana:
    """
    Model description:
    Basic KNN model with 5 classes
    Dataset: mcu13, vinikot, JBL Flip 5 - Auguste - spec_20_20 --> chainsaw, crackling fire, fireworks, gunshot, background
             mcu13, fisher, local speakers - spec_20_20 --> chainsaw, crackling fire, fireworks, gunshot, background
    Dataset équilibré
        - Objectif  : Etudier l'effet d'une PCA sur les performances du KNN et optimiser les hyperparamètres du KNN par GridSearchCV
        - Résultats : le PCA améliore les performances du KNN, avec une meilleure performance autour de 25 composants
        - Conclusion : le PCA est nécessaire pour ce dataset, il améliore les performances du KNN de manière significative.
                       On garde donc un modèle KNN avec PCA, avec une architecture optimisée par GridSearchCV.
                       Meilleure architecture: n_neighbors=1, weights='distance', metric='euclidean', PCA components=25
    Modèle clôturé le jeudi 12 mars 2026 à 00h48
    """
    GS_albaniana = False
    PCA_albaniana = False
    Gen_albaniana = True
    components = [15, 20, 25, 30, 35, 40, 45]

    if PCA_albaniana:
        for x in components:

            records = [("mcu13", "fisher", "local speakers - spec_20_20"),
                    ("mcu13", "vinikot", "JBL Flip 5 - Auguste - spec_20_20")] # 5 classes, 122 samples per class
            data, labels = load_data(records)
            le = LabelEncoder()
            labels_encoded = le.fit_transform(labels)
            data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)


            x_train, x_test, y_train, y_test = train_test_split(data_normalized, labels_encoded, test_size=0.3, random_state=42, shuffle=True, stratify=labels_encoded)

            # Apply PCA for dimensionality reduction if needed (e.g., to 50 components)
            pca = PCA(n_components=x)
            x_train = pca.fit_transform(x_train)
            x_test = pca.transform(x_test)

            if GS_albaniana:
                parameters = {
                    'n_neighbors': [1, 2, 3],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }

                hyperparameter_tuning = GridSearchCV(KNeighborsClassifier(), parameters, cv=5, n_jobs=-1)
                hyperparameter_tuning.fit(x_train, y_train)
                pickle.dump(hyperparameter_tuning, open(f"hyperparameter_tuning albaniana {x} pca.pkl", "wb"))

    if Gen_albaniana:

        # Load data
        records = [("mcu13", "fisher", "local speakers - spec_20_20"),
                    ("mcu13", "vinikot", "JBL Flip 5 - Auguste - spec_20_20")]
        data, labels = load_data(records)

        # Encode labels
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)

        # Normalize data
        data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)

        # Train test split for visualization of results
        x_train, x_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=0.3, random_state=21, shuffle=True, stratify=labels_encoded)

        # Apply PCA for dimensionality reduction for visualization
        pca = PCA(n_components=25)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)

        # Define KNN model
        knn_model = KNeighborsClassifier(n_neighbors=1, weights='distance', metric='euclidean')
        
        # Fit model for visualization and evaluation
        knn_model.fit(x_train, y_train)
        y_pred = knn_model.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        print(f"Test score: {score}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        save_confusion_matrix(knn_model, x_test, y_test, filename="confusion_matrix_albaniana.pdf", show=True)

        # Save model on all the dataset
        pca = PCA(n_components=25)
        data_normalized = pca.fit_transform(data_normalized)
        knn_model.fit(data_normalized, labels)
        pickle.dump(knn_model, open("KNN_albaniana.pkl", "wb"))
        pickle.dump(pca, open("PCA_albaniana.pkl", "wb"))
        print("Model saved as KNN_albaniana.pkl")

# grid_results_files = [f"hyperparameter_tuning albaniana {x} pca.pkl" for x in components]
# all_results = []
# pca_components = components
# for file in grid_results_files:
#     with open(file, "rb") as f:
#         hyperparameter_tuning = pickle.load(f)
#     all_results.append(hyperparameter_tuning.cv_results_)

# sorted_results = []
# for i, results in enumerate(all_results):
#     for mean_score, std_score, params in zip(results['mean_test_score'], results['std_test_score'], results['params']):
#         sorted_results.append((pca_components[i], mean_score, std_score, params))
# # Sort the results by mean_test_score in descending order and write to a markdown file with all pca components results sorted by mean_test_score in descending order

# with open("KNN_albaniana_results.md", "w") as f:
#     f.write("| PCA Components | Mean Test Score | Std Test Score | Neighbors | Metric | Weights |\n")
#     f.write("| --- | --- | --- | --- | --- | --- |\n")
#     sorted_results.sort(key=lambda x: x[1], reverse=True)  # Sort by mean_test_score in descending order
#     for pca_comp, mean_score, std_score, params in sorted_results:
#         f.write(f"| {pca_comp} | {mean_score:.4f} | {std_score:.4f} | {params['n_neighbors']} | {params['metric']} | {params['weights']} |\n")