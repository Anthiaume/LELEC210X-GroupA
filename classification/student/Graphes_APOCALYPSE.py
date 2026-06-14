import matplotlib.pyplot as plt
import pickle

basic_train_losses, basic_valid_losses = pickle.load(open("basic_losses.pkl", "rb"))
normalized_train_losses, normalized_valid_losses = pickle.load(open("normalized_losses.pkl", "rb"))
mstd_train_losses, mstd_val_losses = pickle.load(open("mstd_losses.pkl", "rb"))

x_ticks = range(1, len(basic_train_losses) + 1)

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

axs[0].plot(x_ticks[18:], basic_train_losses[18:], label="Train Loss")
axs[0].plot(x_ticks[18:], basic_valid_losses[18:], label="Validation Loss")
axs[0].set_title("A) Basic MLP Loss Curves")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].legend()

axs[1].plot(x_ticks, normalized_train_losses, label="Train Loss")
axs[1].plot(x_ticks, normalized_valid_losses, label="Validation Loss")
axs[1].set_title("B) Normalized MLP Loss Curves")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Loss")
axs[1].legend()

axs[2].plot(x_ticks, mstd_train_losses, label="Train Loss")
axs[2].plot(x_ticks, mstd_val_losses, label="Validation Loss")
axs[2].set_title("C) Mean and Std MLP Loss Curves")
axs[2].set_xlabel("Epoch")
axs[2].set_ylabel("Loss")
axs[2].legend()

plt.tight_layout()
plt.savefig("loss_curves_comparison_APO.pdf")
