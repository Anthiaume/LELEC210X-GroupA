from student_fct import *
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import torch.nn as nn


# records = [("mcu13", "fisher", "local speakers - spec_20_20"),         # 5 classes, 122 samples per class
#             ("mcu13", "vinikot", "JBL Flip 5 - Auguste - spec_20_20"),
#             ("mcu13", "sud5", "local speakers - spec_20_20"),       # 5 classes, 122 samples per class
#             ("mcu13", "sud11", "local speakers - spec_20_20 - Jonathan"),
#             ("mcu13", "sud11", "local speakers - spec_20_20 - Raphael")]         # 5 classes, 122 samples per class]

# # Load data and labels
# data, labels, labels_encoded, MLP_halikarnassos_classes = load_data(records, classes=["background", "chainsaw", "crackling fire", "fireworks", "gunshot"])

# x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42, shuffle=True, stratify=labels_encoded)  # 0.25 x 0.8 = 0.2
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, shuffle=True, stratify=y_train)  # 0.25 x 0.8 = 0.2


x_train, y_train = pickle.load(open("x_train.pkl", "rb"))
x_val, y_val = pickle.load(open("x_val.pkl", "rb"))

models = load_models()


for model in range(len(models)):

    print(models[model]["model_name"])

    x_train_processed = process_data_for_MLP_for_test(x_train, models[model]["params"])
    x_val_processed   = process_data_for_MLP_for_test(x_val, models[model]["params"])

        # Training the model
    duration = time.time()
    if models[model]["model_name"].split("_")[0].lower() != "fireworks":
        n_epoch = models[model]["params"]["num_epochs"]
    else:
        n_epoch = models[model]["params"]["n_epochs"]
       


    mlp = TorchMLP(hidden_layers_sizes=models[model]["params"]["hidden_layers_sizes"],
                    activation=nn.ReLU,
                    IO=models[model]["params"]["IO"], 
                    num_epochs=n_epoch,
                    batch_size=models[model]["params"]["batch_size"],
                    learning_rate=models[model]["params"]["learning_rate"],
                    dropout_rate=models[model]["params"]["dropout_rate"],
                    class_weights=models[model]["params"]["class_weights"],
                    x_val=x_val_processed, y_val=y_val,
                    plot_loss=False,
                    verbose=True,
                    loss_filename=None)
    
    train_losses, val_losses = mlp.fit(x_train_processed, y_train)

    duration = time.time() - duration
    print(f"Training time: {duration:.2f} seconds")

    pickle.dump((train_losses, val_losses), open(f"train_val_losses_{models[model]['model_name']}.pkl", "wb"))