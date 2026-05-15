# model names: assindria, berolinum, constantia, dispargum, erfordia, francofortum, goslaria, herbipolis, iuliacum, locoritum, misnia, norimberga, patavia, rigomagus, stutgardia



from student_fct import *
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

records = [("mcu13", "fisher", "local speakers - spec_20_20"),         # 5 classes, 122 samples per class
            ("mcu13", "vinikot", "JBL Flip 5 - Auguste - spec_20_20"),
            ("mcu13", "sud5", "local speakers - spec_20_20"),       # 5 classes, 122 samples per class
            ("mcu13", "sud11", "local speakers - spec_20_20 - Jonathan"),
            ("mcu13", "sud11", "local speakers - spec_20_20 - Raphael")]#,
            # ("mcu13", "sud8", "local speakers - spec_20_20")]         # 5 classes, 122 samples per class]

# records = [("mcu13", "sud11", "local speakers - spec_20_20 - Raphael")]         # 5 classes, 122 samples per class]

# Load data and labels
data, labels, labels_encoded, MLP_halikarnassos_classes = load_data(records, classes=["background", "chainsaw", "crackling fire", "fireworks", "gunshot"])
params = {"keeped_frequencies": (0, 19),
        "attenuation_dB_range": None,
        "transform_mean_std": True,
        "n_freq": 20,
        "exposant": 1
}
x_data = process_data_for_KNN_for_test(data, params=params)
print(x_data.shape)
input("Press Enter to continue...")
x_train, x_test, y_train, y_test = train_test_split(x_data, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

for k in range(1, 11):
    
    knn = KNeighborsClassifier(n_neighbors=k, weights="uniform", metric="euclidean")
    knn.fit(x_train, y_train)
    pickle.dump(knn, open(f"knn_model_wtf.pkl", "wb"))
    y_pred_knn = knn.predict(x_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    print(f"KNN Classifier Accuracy: {accuracy_knn:.4f}")

    # show confusion matrix
    conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
    print(f"Confusion Matrix for n neighbors {k}:")
    print(conf_matrix_knn)

# data = data / np.linalg.norm(data, axis=1, keepdims=True)

# pca1 = PCA(n_components=2)
# data_pca = pca1.fit_transform(data)

# plt.figure(figsize=(10, 6))
# for class_label in np.unique(labels_encoded):
#     plt.scatter(data_pca[labels_encoded == class_label, 0], data_pca[labels_encoded == class_label, 1], label=MLP_halikarnassos_classes[class_label], alpha=0.5)
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.title("PCA of Audio Data")
# plt.legend()
# # plt.show()
# plt.close()

# data2 = transform_melspecgram_to_mean_std_data(data, n_freq=20)
# # pca 2
# pca2 = PCA(n_components=2)
# data_pca2 = pca2.fit_transform(data2)
# pickle.dump(pca2, open(f"pca_model_wtf.pkl", "wb"))

# model = MLPClassifier(hidden_layer_sizes=(600, 300, 100), activation='relu', solver='adam', max_iter=500, random_state=42,
#                         early_stopping=True, validation_fraction=0.2, n_iter_no_change=10)
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"MLP Classifier Accuracy: {accuracy:.4f}")

# # show conufsion matrix
# conf_matrix = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:")
# print(conf_matrix)

# for k in range(1, 11):
#     x_train, x_test, y_train, y_test = train_test_split(data2, labels_encoded, test_size=0.2, random_state=3*k, stratify=labels_encoded)
    
#     knn = KNeighborsClassifier(n_neighbors=1, weights="uniform", metric="euclidean")
#     knn.fit(x_train, y_train)
#     pickle.dump(knn, open(f"knn_model_wtf.pkl", "wb"))
#     y_pred_knn = knn.predict(x_test)
#     accuracy_knn = accuracy_score(y_test, y_pred_knn)
#     print(f"KNN Classifier Accuracy: {accuracy_knn:.4f}")

#     # show confusion matrix
#     conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
#     print(f"Confusion Matrix for random state {3*k}:")
#     print(conf_matrix_knn)

# plt.figure(figsize=(10, 6))
# for class_label in np.unique(labels_encoded):
#     plt.scatter(data_pca2[labels_encoded == class_label, 0], data_pca2[labels_encoded == class_label, 1], label=MLP_halikarnassos_classes[class_label], alpha=0.5)
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.title("PCA of Audio Data")
# plt.legend()

# plt.show()

# data3 = transform_melspecgram_to_mean_var_data(data, n_freq=20)
# # pca 3
# pca3 = PCA(n_components=2)
# data_pca3 = pca3.fit_transform(data3)

# plt.figure(figsize=(10, 6))
# for class_label in np.unique(labels_encoded):
#     plt.scatter(data_pca3[labels_encoded == class_label, 0], data_pca3[labels_encoded == class_label, 1], label=MLP_halikarnassos_classes[class_label], alpha=0.5)
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.title("PCA of Audio Data")
# plt.legend()
# plt.show()


# # drop data whose label is not "background"

# data = data[labels == "background"]

# dataLf = keep_frequencies(data, (0, 4))
# dataHf = keep_frequencies(data, (5, 19))

# zerosLF = np.zeros_like(dataHf)
# zerosHF = np.zeros_like(dataLf)

# dataLf = np.concatenate((dataLf, zerosLF), axis=1)

# dataHf = np.concatenate((zerosHF, dataHf), axis=1)

# normLf = np.linalg.norm(dataLf, axis=1, keepdims=True)
# normHf = np.linalg.norm(dataHf, axis=1, keepdims=True)

# calc = normLf**2 + normHf**2

# norm   = np.linalg.norm(data, axis=1, keepdims=True)

# print("LF norm mean:", np.mean(normLf))
# print("HF norm mean:", np.mean(normHf))
# print("calc norm mean:", np.mean(calc))
# print("Normal norm mean:", np.mean(norm))


# models = load_models()

# models.sort(key=lambda x: x["model_name"].split("_")[0]+x["model_name"].split("_")[2])

# models_general = models[8:10]
# models_gunshot = models[10:]
# models[8:10] = models_gunshot
# models[10:] = models_general

# for i in range(len(models)):
#     if "fireworks" in models[i]["model_name"].lower():
#         nepochs = models[i]["params"]["n_epochs"]
#     else:
#         nepochs = models[i]["params"]["num_epochs"]
#     print(f"{i+1}. {models[i]['model_name']}, exposant: {models[i]['params']['exposant']}, dropout_rate: {models[i]['params']['dropout_rate']}, n_epochs: {nepochs}")

# # print(models[0].keys())

# # print(models[0]['params'].keys())
# # print(models[0]["loss_curves"]["train"])
# # print(models[0]["loss_curves"]["val"])

# # num_epochs=models[model]["params"]["num_epochs"],
# # KeyError: 'num_epochs'

# print(models[6]["model_name"])
# print(models[6].keys())
# print(models[6]['params'].keys())
# print(models[6]["params"]["dropout_rate"])
# print(models[6]["epochs"])