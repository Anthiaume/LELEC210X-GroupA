# model names: assindria, berolinum, constantia, dispargum, erfordia, francofortum, goslaria, herbipolis, iuliacum, locoritum, misnia, norimberga, patavia, rigomagus, stutgardia



from student_fct import *
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

records = [("mcu13", "fisher", "local speakers - spec_20_20"),         # 5 classes, 122 samples per class
            ("mcu13", "vinikot", "JBL Flip 5 - Auguste - spec_20_20"),
            ("mcu13", "sud5", "local speakers - spec_20_20"),       # 5 classes, 122 samples per class
            ("mcu13", "sud11", "local speakers - spec_20_20 - Jonathan"),
            ("mcu13", "sud11", "local speakers - spec_20_20 - Raphael")]         # 5 classes, 122 samples per class]

records = [("mcu13", "sud11", "local speakers - spec_20_20 - Raphael")]         # 5 classes, 122 samples per class]

# Load data and labels
data, labels, labels_encoded, MLP_halikarnassos_classes = load_data(records, classes=["background", "chainsaw", "crackling fire", "fireworks", "gunshot"])

data = data / np.linalg.norm(data, axis=1, keepdims=True)

pca1 = PCA(n_components=2)
data_pca = pca1.fit_transform(data)

plt.figure(figsize=(10, 6))
for class_label in np.unique(labels_encoded):
    plt.scatter(data_pca[labels_encoded == class_label, 0], data_pca[labels_encoded == class_label, 1], label=MLP_halikarnassos_classes[class_label], alpha=0.5)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Audio Data")
plt.legend()
plt.show()

data2 = transform_melspecgram_to_mean_std_data(data, n_freq=20)
# pca 2
pca2 = PCA(n_components=2)
data_pca2 = pca2.fit_transform(data2)

plt.figure(figsize=(10, 6))
for class_label in np.unique(labels_encoded):
    plt.scatter(data_pca2[labels_encoded == class_label, 0], data_pca2[labels_encoded == class_label, 1], label=MLP_halikarnassos_classes[class_label], alpha=0.5)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Audio Data")
plt.legend()
plt.show()

data3 = transform_melspecgram_to_mean_var_data(data, n_freq=20)
# pca 3
pca3 = PCA(n_components=2)
data_pca3 = pca3.fit_transform(data3)

plt.figure(figsize=(10, 6))
for class_label in np.unique(labels_encoded):
    plt.scatter(data_pca3[labels_encoded == class_label, 0], data_pca3[labels_encoded == class_label, 1], label=MLP_halikarnassos_classes[class_label], alpha=0.5)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Audio Data")
plt.legend()
plt.show()


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