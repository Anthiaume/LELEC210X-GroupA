import os
from os import listdir
from os.path import isfile, join
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import time

def load_data(records):
    """
    Inputs:
    - records : list of tuples (mcu, local, speaker) specifying the data to load. For example:
        Example : records = [("mcu13", "vinikot", "JBL Flip 5 - Auguste - spec_20_20"),
                             ("mcu13", "fisher", "local speakers - spec_20_20")       ]
    Output:
    - data: numpy array of melspectrograms
    - labels: numpy array of labels (e.g. "chainsaw", "crackling fire", "fireworks", "gunshot", "background")
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
        files_in_directory = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith(".pkl"))]
        for file in files_in_directory:
            with open (os.path.join(mypath, file), "rb") as f:
                data.append(pickle.load(f))
                labels.append(file.split("_")[0])
                
    data = np.array(data)
    labels = np.array(labels)

    return data, labels

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

class TorchMLP(nn.Module):
    def __init__(self, hidden_layers_sizes=[300, 300, 200, 100, 50], activation=nn.ReLU, IO = (400, 5), num_epochs=500, batch_size=None,
                 plot_loss=False, loss_filename=None, verbose=True, x_val=None, y_val=None, learning_rate=1e-6):

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
        self.num_classes = IO[1]

        # Create a sequential model
        self.model = nn.Sequential()

        # Create the hidden layers and add them to the model
        for layer in range(len(hidden_layers_sizes)+1):

            # Add the input layer and the first hidden layer
            if layer == 0:
                self.model.add_module("input", nn.Linear(IO[0], hidden_layers_sizes[0]))
            
            # Add hidden layers and activation functions
            if layer < len(hidden_layers_sizes):
                self.model.add_module(f"act{layer+1}", activation())
                self.model.add_module(f"dense{layer+1}", nn.Linear(hidden_layers_sizes[layer], (hidden_layers_sizes[layer+1] if layer+1 < len(hidden_layers_sizes) else IO[1])))

    def forward(self, x):
        """
        This method defines the forward pass of the model. It takes an input tensor x and returns the output of the model (logits).
        Note: The softmax activation is not applied here because we will use CrossEntropyLoss which expects raw logits.
        Inputs - x: input tensor of shape (batch_size, input_features)
        Outputs - logits: output tensor of shape (batch_size, num_classes)
        """
        return self.model(x)

    def predict_proba(self, x):
        """
        This method applies the softmax function to the output of the forward pass to get probabilities for each class.
        Inputs - x: input tensor of shape (batch_size, input_features)
        Outputs - probs: output tensor of shape (batch_size, num_classes) where each row sums to
                  1 and represents the predicted probabilities for each class.
        """
        return torch.softmax(self(x), dim=1)
    
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
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

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
            plt.tight_layout()
            if self.loss_filename is not None:
                plt.savefig(self.loss_filename, bbox_inches='tight')
            if self.plot_loss:
                plt.show()
         
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
        plt.savefig(filename, bbox_inches='tight')
        if show:
            plt.show()

    def GridSearch(self, x_train, y_train, param_grid, cv=5, verbose=True):
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
        print(n_samples)

        # Placeholder for best parameters and best score
        best_params = None
        best_score  = -1.0
        results     = []

        # Generate all combinations of hyperparameters
        from itertools import product
        keys, values = zip(*param_grid.items())
        all_combinations = [dict(zip(keys, combo)) for combo in product(*values)]
        n_combinations = len(all_combinations)

        # Precompute CV fold indices (stratified by shuffling)
        indices = np.random.permutation(n_samples)
        folds = np.array_split(indices, cv)  # cv roughly equal folds

        if verbose:
            print(f"Grid search over {n_combinations} combinations × {cv} folds "
                f"= {n_combinations * cv} trainings\n")

        for i, params in enumerate(all_combinations):

            fold_scores = []

            for fold_idx in range(cv):
                print("Current advance:", i+1, "/", n_combinations, "fold:", fold_idx+1, "/", cv)
                # --- Build train/val split for this fold ---
                val_indices   = folds[fold_idx]
                train_indices = np.concatenate([folds[j] for j in range(cv) if j != fold_idx])

                x_fold_train = x_train[train_indices]
                y_fold_train = y_train[train_indices]
                x_fold_val   = x_train[val_indices]
                y_fold_val   = y_train[val_indices]

                # --- Instantiate a fresh model with current params ---
                # Separate IO from the rest since it's a constructor param but not a hyperparameter
                # that changes between folds
                model = TorchMLP(**params)
                model.fit(x_fold_train, y_fold_train)

                fold_score = model.score(x_fold_val, y_fold_val)
                fold_scores.append(fold_score)

            mean_score = np.mean(fold_scores)
            std_score  = np.std(fold_scores)

            results.append({
                "params":     params,
                "mean_score": mean_score,
                "std_score":  std_score,
                "fold_scores": fold_scores
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
        pickle.dump(results, open(f"ephesos_GS_{time.ctime().replace(":", "_")}.pkl", "wb"))
        return best_params, results