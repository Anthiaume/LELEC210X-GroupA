from load_data import load_data
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
#import principal component analysis (PCA) for dimensionality reduction if needed
from sklearn.decomposition import PCA
# model names: albaniana, blariacum, carvium, daventria, elenio, fectio, ganuenta, horgana, matilo, nigrum, praetorium, sablones, tablis, ulpia, venlonum

KNN_albaniana = 0
if KNN_albaniana:
    """
    Model description:
    Basic KNN model with 5 classes
    """
    GS_albaniana = True

    for x in [2, 5, 10, 20, 50]:
        records = [("mcu13", "fisher", "local speakers - spec_20_20")] # 5 classes, 122 samples per class
        data, labels = load_data(records)
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)


        x_train, x_test, y_train, y_test = train_test_split(data_normalized, labels_encoded, test_size=0.3, random_state=42, shuffle=True, stratify=labels_encoded)

        # Apply PCA for dimensionality reduction if needed (e.g., to 50 components)
        pca = PCA(n_components=x)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)

        records_2 = [("mcu13", "vinikot", "JBL Flip 5 - Auguste - spec_20_20")] # 4 classes
        data_2, labels_2 = load_data(records_2)
        data_2_normalized = data_2 / np.linalg.norm(data_2, axis=1, keepdims=True)


        if GS_albaniana:
            parameters = {
                'n_neighbors': [1, 2, 3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }

            hyperparameter_tuning = GridSearchCV(KNeighborsClassifier(), parameters, cv=5, n_jobs=-1)
            hyperparameter_tuning.fit(x_train, y_train)
            pickle.dump(hyperparameter_tuning, open(f"hyperparameter_tuning albaniana {x} pca.pkl", "wb"))

grid_results_files = [f"hyperparameter_tuning albaniana {x} pca.pkl" for x in [2, 5, 10, 20, 50]]
all_results = []
pca_components = [2, 5, 10, 20, 50]
for file in grid_results_files:
    with open(file, "rb") as f:
        hyperparameter_tuning = pickle.load(f)
    all_results.append(hyperparameter_tuning.cv_results_)

sorted_results = []
for i, results in enumerate(all_results):
    for mean_score, std_score, params in zip(results['mean_test_score'], results['std_test_score'], results['params']):
        sorted_results.append((pca_components[i], mean_score, std_score, params))
# Sort the results by mean_test_score in descending order and write to a markdown file with all pca components results sorted by mean_test_score in descending order

with open("KNN_albaniana_results.md", "w") as f:
    f.write("| PCA Components | Mean Test Score | Std Test Score | Neighbors | Metric | Weights |\n")
    f.write("| --- | --- | --- | --- | --- | --- |\n")
    sorted_results.sort(key=lambda x: x[1], reverse=True)  # Sort by mean_test_score in descending order
    for pca_comp, mean_score, std_score, params in sorted_results:
        f.write(f"| {pca_comp} | {mean_score:.4f} | {std_score:.4f} | {params['n_neighbors']} | {params['metric']} | {params['weights']} |\n")