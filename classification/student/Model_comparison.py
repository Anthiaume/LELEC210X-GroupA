from student_fct import load_data, load_compacted_data, save_confusion_matrix_from_array, add_background, suppress_low_frequencies, TorchMLP
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


records = [("mcu13", "sud11", "local speakers - spec_20_20 - Demo 3")]         # 5 classes, 122 samples per class]

# Load data and labels
data, labels = load_data(records)
labels[labels == "voices"] = "background"  # Rename "voices" to "background"

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
MLP_gangra_classes = le.classes_

def load_data_gangra(n_melvecs_to_suppress=10, bool_add_background=False, labels_encoded=labels_encoded):
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
    else:
        labels_e = labels_encoded
    return data_normalized_HF, data_HF_normalized, labels_e

data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)
data_normalized_f = data_normalized**1
data_normalized_HF_gangra, _, labels_g = load_data_gangra(n_melvecs_to_suppress=7, bool_add_background=False)
data_normalized_HF_gangra = data_normalized_HF_gangra**0.6

with open("MLP_ephesos_pytorch.pkl", "rb") as f:
    MLP_ephesos_pytorch = pickle.load(f)

with open("MLP_flaviopolis_pytorch.pkl", "rb") as f:
    MLP_flaviopolis_pytorch = pickle.load(f)

with open("MLP_gangra_pytorch.pkl", "rb") as f:
    MLP_gangra_pytorch = pickle.load(f)

# Evaluate MLP_ephesos_pytorch on the data
predictions_ephesos = MLP_ephesos_pytorch.predict(data_normalized)
# Evaluate MLP_flaviopolis_pytorch on the data
predictions_flaviopolis = MLP_flaviopolis_pytorch.predict(data_normalized_f)
# Evaluate MLP_gangra_pytorch on the data
predictions_gangra = MLP_gangra_pytorch.predict(data_normalized_HF_gangra)

# Accuracy scores
accuracy_ephesos = accuracy_score(labels_encoded, predictions_ephesos)
accuracy_flaviopolis = accuracy_score(labels_encoded, predictions_flaviopolis)
accuracy_gangra = accuracy_score(labels_g, predictions_gangra)

print(f"Accuracy of MLP_ephesos_pytorch: {accuracy_ephesos:.4f}")
print(f"Accuracy of MLP_flaviopolis_pytorch: {accuracy_flaviopolis:.4f}")
print(f"Accuracy of MLP_gangra_pytorch: {accuracy_gangra:.4f}")

# Save confusion matrices
# save_confusion_matrix_from_array(confusion_matrix=confusion_matrix(labels_encoded, predictions_ephesos), filename="confusion_matrix_ephesos_Demo3.pdf", show=True, labels=MLP_gangra_classes)
# save_confusion_matrix_from_array(confusion_matrix=confusion_matrix(labels_encoded, predictions_flaviopolis), filename="confusion_matrix_flaviopolis_Demo3.pdf", show=True, labels=MLP_gangra_classes)
# save_confusion_matrix_from_array(confusion_matrix=confusion_matrix(labels_g, predictions_gangra), filename="confusion_matrix_gangra_Demo3.pdf", show=True, labels=MLP_gangra_classes)