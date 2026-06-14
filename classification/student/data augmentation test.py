import numpy as np
import matplotlib.pyplot as plt
from student_fct import *

records = [("mcu13", "fisher", "local speakers - spec_20_20"), ("mcu13", "vinikot", "JBL Flip 5 - Auguste - spec_20_20"), ("mcu13", "sud5", "local speakers - spec_20_20"), ("mcu13", "sud11", "local speakers - spec_20_20 - Jonathan"), ("mcu13", "sud11", "local speakers - spec_20_20 - Raphael")]#,

data, labels, labels_encoded, classes = load_data(records, classes=["background", "chainsaw", "crackling fire", "fireworks", "gunshot"])

x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

batch_size = len(x_train)

def create_mlp(IO, loss_filename, x_val=x_test, y_val=y_test):
    mlp = TorchMLP(hidden_layers_sizes=(600, 300, 100),
                activation=nn.ReLU,
                IO=IO, 
                num_epochs=400,
                batch_size=batch_size,
                learning_rate=1e-3,
                dropout_rate=0.5,
                x_val=x_val, y_val=y_val,
                plot_loss=True,
                verbose=False,
                loss_filename=loss_filename)
    return mlp


basic = 1
if basic:
    mlp = create_mlp(IO=(x_train.shape[1], len(classes)), loss_filename="loss_basic.pdf", x_val=x_test, y_val=y_test)
    basic_train_losses, basic_valid_losses = mlp.fit(x_train, y_train)

    accuracy_val = mlp.score(x_test, y_test)
    print(f"MLP Classifier basic Accuracy: {accuracy_val:.4f}")

# Normalize data
x_train_normalized = x_train / np.linalg.norm(x_train, axis=1, keepdims=True)
x_test_normalized = x_test / np.linalg.norm(x_test, axis=1, keepdims=True)
# print("Basic MLP Accuracy:", 60, "%")

normalize = 1
if normalize:
    mlp_normalized = create_mlp(IO=(x_train_normalized.shape[1], len(classes)), loss_filename="loss_normalized.pdf", x_val=x_test_normalized, y_val=y_test)
    normalized_train_losses, normalized_valid_losses = mlp_normalized.fit(x_train_normalized, y_train)

    accuracy_val_normalized = mlp_normalized.score(x_test_normalized, y_test)
    print(f"MLP Classifier Accuracy (justNormalized): {accuracy_val_normalized:.4f}")
# print("Basic MLP Accuracy with normalization:", 80, "%")


x_train_N_B, _ = add_background(data_normalized=x_train_normalized, labels=y_train, attenuation_dB_range=(-20, 0))
x_test_N_B, _   = add_background(data_normalized=x_test_normalized, labels=y_test, attenuation_dB_range=(-20, 0))

background = 0
if background:
    mlp_background = create_mlp(IO=(x_train_N_B.shape[1], len(classes)), loss_filename="loss_background.pdf", x_val=x_test_N_B, y_val=y_test)
    mlp_background.fit(x_train_N_B, y_train)

    accuracy_val_background = mlp_background.score(x_test_N_B, y_test)
    print(f"MLP Classifier Accuracy (Background): {accuracy_val_background:.4f}")

# print("Basic MLP Accuracy with background noise:", 69, "%")

x_train_HF = keep_frequencies(data=x_train, n_melvecs_to_keep=(5,19))
x_test_HF  = keep_frequencies(data=x_test,  n_melvecs_to_keep=(5,19))
x_train_HF_normalized = x_train_HF / (np.linalg.norm(x_train_HF, axis=1, keepdims=True) + 1e-16)
x_test_HF_normalized = x_test_HF / (np.linalg.norm(x_test_HF, axis=1, keepdims=True) + 1e-16)
x_train_HF_N_B, _ = add_background(data_normalized=x_train_HF_normalized, labels=y_train, attenuation_dB_range=(-20, 0))
x_test_HF_N_B, _   = add_background(data_normalized=x_test_HF_normalized, labels=y_test, attenuation_dB_range=(-20, 0))


high_freq = 0
if high_freq:
    # print("Training MLP on high frequencies...", x_train_HF.shape[1], len(classes))

    mlp_high_freq_N_B = create_mlp(IO=(x_train_HF_N_B.shape[1], len(classes)), loss_filename="loss_high_freq_N_B.pdf", x_val=x_test_HF_N_B, y_val=y_test)
    mlp_high_freq_N_B.fit(x_train_HF_N_B, y_train)
    accuracy_val_high_freq_N_B = mlp_high_freq_N_B.score(x_test_HF_N_B, y_test)
    print(f"MLP Classifier Accuracy (High Frequencies with Background): {accuracy_val_high_freq_N_B:.4f}")

    mlp_high_freq_normalized = create_mlp(IO=(x_train_HF_normalized.shape[1], len(classes)), loss_filename="loss_high_freq_normalized.pdf", x_val=x_test_HF_normalized, y_val=y_test)
    mlp_high_freq_normalized.fit(x_train_HF_normalized, y_train)
    accuracy_val_high_freq_normalized = mlp_high_freq_normalized.score(x_test_HF_normalized, y_test)
    print(f"MLP Classifier Accuracy (High Frequencies Normalized): {accuracy_val_high_freq_normalized:.4f}")

# print("MLP Classifier Accuracy (High Frequencies):", 45, "%")
# print("MLP Classifier Accuracy (High Frequencies with Background):", 67, "%")
# print("MLP Classifier Accuracy (High Frequencies Normalized):", 73.85, "%")

x_train_N_mstd = transform_melspecgram_to_mean_std_data(x_train_normalized, n_freq=20)
x_test_N_mstd = transform_melspecgram_to_mean_std_data(x_test_normalized, n_freq=20)

x_train_N_B_mstd = transform_melspecgram_to_mean_std_data(x_train_N_B, n_freq=20)
x_test_N_B_mstd = transform_melspecgram_to_mean_std_data(x_test_N_B, n_freq=20)

x_train_HF_N_mstd = transform_melspecgram_to_mean_std_data(x_train_HF_normalized, n_freq=15)
x_test_HF_N_mstd = transform_melspecgram_to_mean_std_data(x_test_HF_normalized, n_freq=15)

x_train_HF_N_B_mstd = transform_melspecgram_to_mean_std_data(x_train_HF_N_B, n_freq=15)
x_test_HF_N_B_mstd = transform_melspecgram_to_mean_std_data(x_test_HF_N_B, n_freq=15)

mean_std = 1
if mean_std:
    # 1. MLP on mean and std of normalized data
    mlp_mean_std = create_mlp(IO=(x_train_N_mstd.shape[1], len(classes)), loss_filename="loss_mean_std.pdf", x_val=x_test_N_mstd, y_val=y_test)
    mstd_train_losses, mstd_val_losses = mlp_mean_std.fit(x_train_N_mstd, y_train)
    accuracy_val_mean_std = mlp_mean_std.score(x_test_N_mstd, y_test)
    print(f"MLP Classifier Accuracy (Mean and Std of Normalized Data): {accuracy_val_mean_std:.4f}")

    # # 2. MLP on mean and std of normalized data with background
    # mlp_mean_std_N_B = create_mlp(IO=(x_train_N_B_mstd.shape[1], len(classes)), loss_filename="loss_mean_std_N_B.pdf", x_val=x_test_N_B_mstd, y_val=y_test)
    # mlp_mean_std_N_B.fit(x_train_N_B_mstd, y_train)
    # accuracy_val_mean_std_N_B = mlp_mean_std_N_B.score(x_test_N_B_mstd, y_test)
    # print(f"MLP Classifier Accuracy (Mean and Std of Normalized Data with Background): {accuracy_val_mean_std_N_B:.4f}")

    # # 3. MLP on mean and std of high frequencies normalized data
    # mlp_mean_std_HF = create_mlp(IO=(x_train_HF_N_mstd.shape[1], len(classes)), loss_filename="loss_mean_std_HF.pdf", x_val=x_test_HF_N_mstd, y_val=y_test)
    # mlp_mean_std_HF.fit(x_train_HF_N_mstd, y_train)
    # accuracy_val_mean_std_HF = mlp_mean_std_HF.score(x_test_HF_N_mstd, y_test)
    # print(f"MLP Classifier Accuracy (Mean and Std of High Frequencies Normalized Data): {accuracy_val_mean_std_HF:.4f}")

    # # 4. MLP on mean and std of high frequencies normalized data with background
    # mlp_mean_std_HF_N_B = create_mlp(IO=(x_train_HF_N_B_mstd.shape[1], len(classes)), loss_filename="loss_mean_std_HF_N_B.pdf", x_val=x_test_HF_N_B_mstd, y_val=y_test)
    # mlp_mean_std_HF_N_B.fit(x_train_HF_N_B_mstd, y_train)
    # accuracy_val_mean_std_HF_N_B = mlp_mean_std_HF_N_B.score(x_test_HF_N_B_mstd, y_test)
    # print(f"MLP Classifier Accuracy (Mean and Std of High Frequencies Normalized Data with Background): {accuracy_val_mean_std_HF_N_B:.4f}")




# print("MLP Classifier Accuracy (Mean and Std of Normalized Data): 0.7556")
# print("MLP Classifier Accuracy (Mean and Std of Normalized Data with Background): 0.6889")
# print("MLP Classifier Accuracy (Mean and Std of High Frequencies Normalized Data): 0.7470")
# print("MLP Classifier Accuracy (Mean and Std of High Frequencies Normalized Data with Background): 0.6632")

x_train_normalized_exp05 = x_train_normalized ** 0.5
x_test_normalized_exp05 = x_test_normalized ** 0.5

x_train_N_B_exp05 = x_train_N_B ** 0.5
x_test_N_B_exp05 = x_test_N_B ** 0.5

x_train_HF_normalized_exp05 = x_train_HF_normalized ** 0.5
x_test_HF_normalized_exp05 = x_test_HF_normalized ** 0.5

x_train_HF_N_B_exp05 = x_train_HF_N_B ** 0.5
x_test_HF_N_B_exp05 = x_test_HF_N_B ** 0.5

x_train_N_B_mstd_exp05 = x_train_N_B_mstd ** 0.5
x_test_N_B_mstd_exp05 = x_test_N_B_mstd ** 0.5

x_train_N_mstd_exp05 = x_train_N_mstd ** 0.5
x_test_N_mstd_exp05 = x_test_N_mstd ** 0.5

x_train_HF_N_mstd_exp05 = x_train_HF_N_mstd ** 0.5
x_test_HF_N_mstd_exp05 = x_test_HF_N_mstd ** 0.5

x_train_HF_N_B_mstd_exp05 = x_train_HF_N_B_mstd ** 0.5
x_test_HF_N_B_mstd_exp05 = x_test_HF_N_B_mstd ** 0.5

expo = 0
if expo:
    mlp_exp05 = create_mlp(IO=(x_train_normalized_exp05.shape[1], len(classes)), loss_filename="loss_exp05.pdf", x_val=x_test_normalized_exp05, y_val=y_test)
    mlp_exp05.fit(x_train_normalized_exp05, y_train)
    accuracy_val_exp05 = mlp_exp05.score(x_test_normalized_exp05, y_test)
    print(f"MLP Classifier Accuracy (Normalized Data with Exponent 0.5): {accuracy_val_exp05:.4f}")

    mlp_N_B_exp05 = create_mlp(IO=(x_train_N_B_exp05.shape[1], len(classes)), loss_filename="loss_N_B_exp05.pdf", x_val=x_test_N_B_exp05, y_val=y_test)
    mlp_N_B_exp05.fit(x_train_N_B_exp05, y_train)
    accuracy_val_N_B_exp05 = mlp_N_B_exp05.score(x_test_N_B_exp05, y_test)
    print(f"MLP Classifier Accuracy (Normalized Data with Background and Exponent 0.5): {accuracy_val_N_B_exp05:.4f}")

    mlp_HF_normalized_exp05 = create_mlp(IO=(x_train_HF_normalized_exp05.shape[1], len(classes)), loss_filename="loss_HF_normalized_exp05.pdf", x_val=x_test_HF_normalized_exp05, y_val=y_test)
    mlp_HF_normalized_exp05.fit(x_train_HF_normalized_exp05, y_train)
    accuracy_val_HF_normalized_exp05 = mlp_HF_normalized_exp05.score(x_test_HF_normalized_exp05, y_test)
    print(f"MLP Classifier Accuracy (High Frequencies Normalized Data with Exponent 0.5): {accuracy_val_HF_normalized_exp05:.4f}")

    mlp_HF_N_B_exp05 = create_mlp(IO=(x_train_HF_N_B_exp05.shape[1], len(classes)), loss_filename="loss_HF_N_B_exp05.pdf", x_val=x_test_HF_N_B_exp05, y_val=y_test)
    mlp_HF_N_B_exp05.fit(x_train_HF_N_B_exp05, y_train)
    accuracy_val_HF_N_B_exp05 = mlp_HF_N_B_exp05.score(x_test_HF_N_B_exp05, y_test)
    print(f"MLP Classifier Accuracy (High Frequencies Normalized Data with Background and Exponent 0.5): {accuracy_val_HF_N_B_exp05:.4f}")

    mlp_N_B_mstd_exp05 = create_mlp(IO=(x_train_N_B_mstd_exp05.shape[1], len(classes)), loss_filename="loss_N_B_mstd_exp05.pdf", x_val=x_test_N_B_mstd_exp05, y_val=y_test)
    mlp_N_B_mstd_exp05.fit(x_train_N_B_mstd_exp05, y_train)
    accuracy_val_N_B_mstd_exp05 = mlp_N_B_mstd_exp05.score(x_test_N_B_mstd_exp05, y_test)
    print(f"MLP Classifier Accuracy (Mean and Std of Normalized Data with Background and Exponent 0.5): {accuracy_val_N_B_mstd_exp05:.4f}")

    mlp_N_mstd_exp05 = create_mlp(IO=(x_train_N_mstd_exp05.shape[1], len(classes)), loss_filename="loss_N_mstd_exp05.pdf", x_val=x_test_N_mstd_exp05, y_val=y_test)
    mlp_N_mstd_exp05.fit(x_train_N_mstd_exp05, y_train)
    accuracy_val_N_mstd_exp05 = mlp_N_mstd_exp05.score(x_test_N_mstd_exp05, y_test)
    print(f"MLP Classifier Accuracy (Mean and Std of Normalized Data with Exponent 0.5): {accuracy_val_N_mstd_exp05:.4f}")

    mlp_HF_N_mstd_exp05 = create_mlp(IO=(x_train_HF_N_mstd_exp05.shape[1], len(classes)), loss_filename="loss_HF_N_mstd_exp05.pdf", x_val=x_test_HF_N_mstd_exp05, y_val=y_test)
    mlp_HF_N_mstd_exp05.fit(x_train_HF_N_mstd_exp05, y_train)
    accuracy_val_HF_N_mstd_exp05 = mlp_HF_N_mstd_exp05.score(x_test_HF_N_mstd_exp05, y_test)
    print(f"MLP Classifier Accuracy (Mean and Std of High Frequencies Normalized Data with Exponent 0.5): {accuracy_val_HF_N_mstd_exp05:.4f}")

    mlp_HF_N_B_mstd_exp05 = create_mlp(IO=(x_train_HF_N_B_mstd_exp05.shape[1], len(classes)), loss_filename="loss_HF_N_B_mstd_exp05.pdf", x_val=x_test_HF_N_B_mstd_exp05, y_val=y_test)
    mlp_HF_N_B_mstd_exp05.fit(x_train_HF_N_B_mstd_exp05, y_train)
    accuracy_val_HF_N_B_mstd_exp05 = mlp_HF_N_B_mstd_exp05.score(x_test_HF_N_B_mstd_exp05, y_test)
    print(f"MLP Classifier Accuracy (Mean and Std of High Frequencies Normalized Data with Background and Exponent 0.5): {accuracy_val_HF_N_B_mstd_exp05:.4f}")







# MLP Classifier basic Accuracy: 0.2000
# MLP Classifier Accuracy (justNormalized): 0.8068
# MLP Classifier Accuracy (Background): 0.7162
# MLP Classifier Accuracy (High Frequencies with Background): 0.6923
# MLP Classifier Accuracy (High Frequencies Normalized): 0.7453
# MLP Classifier Accuracy (Mean and Std of Normalized Data): 0.8684
# MLP Classifier Accuracy (Mean and Std of Normalized Data with Background): 0.7778
# MLP Classifier Accuracy (Mean and Std of High Frequencies Normalized Data): 0.8427
# MLP Classifier Accuracy (Mean and Std of High Frequencies Normalized Data with Background): 0.7487
# MLP Classifier Accuracy (Normalized Data with Exponent 0.5): 0.7983
# MLP Classifier Accuracy (Normalized Data with Background and Exponent 0.5): 0.6188
# MLP Classifier Accuracy (High Frequencies Normalized Data with Exponent 0.5): 0.7863
# MLP Classifier Accuracy (High Frequencies Normalized Data with Background and Exponent 0.5): 0.5419
# MLP Classifier Accuracy (Mean and Std of Normalized Data with Background and Exponent 0.5): 0.7761
# MLP Classifier Accuracy (Mean and Std of Normalized Data with Exponent 0.5): 0.8889
# MLP Classifier Accuracy (Mean and Std of High Frequencies Normalized Data with Exponent 0.5): 0.8496
# MLP Classifier Accuracy (Mean and Std of High Frequencies Normalized Data with Background and Exponent 0.5): 0.7333

# Triplot on one row to show all the loss curves together
#save losses

pickle.dump((basic_train_losses, basic_valid_losses), open("basic_losses.pkl", "wb"))
pickle.dump((normalized_train_losses, normalized_valid_losses), open("normalized_losses.pkl", "wb"))
pickle.dump((mstd_train_losses, mstd_val_losses), open("mstd_losses.pkl", "wb"))

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

axs[0].plot(basic_train_losses, label="Train Loss")
axs[0].plot(basic_valid_losses, label="Validation Loss")
axs[0].set_title("Basic MLP Loss Curves")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].legend()

axs[1].plot(normalized_train_losses, label="Train Loss")
axs[1].plot(normalized_valid_losses, label="Validation Loss")
axs[1].set_title("Normalized MLP Loss Curves")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Loss")
axs[1].legend()

axs[2].plot(mstd_train_losses, label="Train Loss")
axs[2].plot(mstd_val_losses, label="Validation Loss")
axs[2].set_title("Mean and Std MLP Loss Curves")
axs[2].set_xlabel("Epoch")
axs[2].set_ylabel("Loss")
axs[2].legend()

plt.tight_layout()
plt.savefig("loss_curves_comparison_APO.pdf")
plt.show()