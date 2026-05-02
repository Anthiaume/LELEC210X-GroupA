# import matplotlib.pyplot as plt
# import subprocess
# import warnings
# import numpy as np

# warnings.filterwarnings("ignore")

# # Define the filename and the real class for analysis
# filename= ["results_uart_gunshot_25.txt", "results_uart_gunshot_50.txt", "results_uart_gunshot_100.txt"]
# filename = ["results_uart_gunshot_25.txt"]
# filename= ["results_uart_gunshot_25.txt", "results_uart_gunshot_50.txt", "results_uart_gunshot_100.txt"]

# real_class = "gunshot" # Change this to the real class you want to analyze
# fontsize = 12

# # Read the data from the file
# data = []
# for file in filename:
#     data.append(open(file, 'r').readlines())

# # Extract model names and predictions from the lines
# models = []
# predictions = []
# for file_data in data:
#     models.append([line.strip().split(" ")[1:][0] for line in file_data if (line.strip() != "New predictions :" and line.strip() != "")])
#     predictions.append([line.strip().split(" ")[-1][1:] for line in file_data if (line.strip() != "New predictions :" and line.strip() != "")])

# # Count the predictions for each model
# count_dict = {}
# for file in range(len(filename)):
#     count_dict[filename[file]] = {}
#     for model, prediction in zip(models[file], predictions[file]):
#         if model not in count_dict[filename[file]]:
#             count_dict[filename[file]][model] = {
#                 "background": 0,
#                 "chainsaw": 0,
#                 "fire": 0,
#                 "fireworks": 0,
#                 "gunshot": 0
#             }
#         count_dict[filename[file]][model][prediction] += 1

# # Normalize the counts to get percentages
# norm_count_dict = {}
# for file in range(len(filename)):
#     norm_count_dict[filename[file]] = {}
#     for model, counts in count_dict[filename[file]].items():
#         total = sum(counts.values())
#         norm_count_dict[filename[file]][model] = {cls: (count / total) * 100 for cls, count in counts.items()}

# # Rename the models for better visualization
# norm_count_dict_renamed = {}
# for file in range(len(filename)):
#     norm_count_dict_renamed[filename[file]] = {}
#     for model in sorted(norm_count_dict[filename[file]].keys()):
#         model_class = model.split("_")[0].upper()
#         freq_model = "HF" if "HF" in model else "AF"
#         new_name = f"{model_class} - {freq_model}"
#         norm_count_dict_renamed[filename[file]][new_name] = norm_count_dict[filename[file]].pop(model)


# # Create subplots for each model
# fig, axes = plt.subplots(2, 6, figsize=(20, 10))
# models = ["general", "background", "chainsaw", "fire", "fireworks", "gunshot"]
# indexes = [0, 1, 2, 3, 4, 5]
# colors = ["blue", "orange", "green", "red", "purple", "brown"]

# # for each model, plot the triple of bars for AF and HF in function of loud intensity
# # norm_count_dict_renamed[file] is a dict with keys as model names and values as dict of class counts in percentage

# new_dict = {}
# for file in range(len(filename)):
#     for model, counts in norm_count_dict_renamed[filename[file]].items():
#         if model not in new_dict:
#             new_dict[model] = counts
#             for i in counts:
#                 new_dict[model][i] = [counts[i]]
#         else:
#             for i in counts:
#                 new_dict[model][i].append(counts[i])

# # for i, (model, counts) in enumerate(new_dict.items()):

# #     column = models.index(model.split(" - ")[0].lower())
# #     row = 1 if "HF" in model else 0
# #     color = colors[column]
# #     print(counts.keys(), counts.values())
# #     axes[row, column].bar(counts.keys(), counts.values())#, color=color)
# #     axes[row, column].set_title(model, fontsize=1.75*fontsize)
# #     # axes[row, column].set_xlabel("Predicted Class", fontsize=1.5*fontsize)
# #     axes[row, column].set_ylabel("Count (%)", fontsize=1.5*fontsize)
# #     axes[row, column].set_xticklabels(counts.keys(), rotation=45, fontsize=fontsize)
# #     axes[row, column].set_yticklabels(axes[row, column].get_yticklabels(), fontsize=fontsize)
# # plt.tight_layout()
# # plt.savefig(f"predictions_analysis_{filename.split('_uart_')[1].split('.')[0]}.pdf")

# # # Open the PDF file using the default PDF viewer
# # subprocess.Popen(f"predictions_analysis_{filename.split('_uart_')[1].split('.')[0]}.pdf", shell=True)


# # Plot
# model_columns = ["general", "background", "chainsaw", "fire", "fireworks", "gunshot"]
# fig, axes = plt.subplots(2, 6, figsize=(24, 10))

# x = np.arange(len(classes))          # 5 positions (one per predicted class)
# n_bars = len(filename)                # 3 bars per position (one per intensity)
# width = 0.25                          # width of each bar
# colors_intensity = ["steelblue", "darkorange", "seagreen"]

# for model_name, counts in new_dict.items():
#     col = model_columns.index(model_name.split(" - ")[0].lower())
#     row = 1 if "HF" in model_name else 0
#     ax = axes[row, col]

#     for i, (intensity_label, color) in enumerate(zip(labels_intensity, colors_intensity)):
#         values = [counts[cls][i] for cls in classes]
#         ax.bar(x + i * width, values, width, label=intensity_label, color=color)

#     ax.set_title(model_name, fontsize=1.5 * fontsize)
#     ax.set_ylabel("Count (%)", fontsize=fontsize)
#     ax.set_xticks(x + width)  # centre les labels sous les groupes
#     ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=fontsize - 1)
#     ax.legend(title="Intensity", fontsize=fontsize - 2)

# plt.tight_layout()
# plt.savefig("predictions_analysis_gunshot.pdf")

import subprocess
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

filename = ["results_uart_gunshot_25.txt", "results_uart_gunshot_50.txt", "results_uart_gunshot_100.txt"]
filename = ["results_uart_background_25.txt", "results_uart_background_50.txt", "results_uart_background_100.txt"]
filename = ["results_uart_chainsaw_25.txt", "results_uart_chainsaw_50.txt", "results_uart_chainsaw_100.txt"]
filename = ["results_uart_fire_25.txt", "results_uart_fire_50.txt", "results_uart_fire_100.txt"]
filename = ["results_uart_fireworks_25.txt", "results_uart_fireworks_50.txt", "results_uart_fireworks_100.txt"]

labels_intensity = ["25%", "50%", "100%"]
real_class = "gunshot"
fontsize = 12

# Read the data
data = []
for file in filename:
    data.append(open(file, 'r').readlines())

# Extract model names and predictions
models_raw = []
predictions_raw = []
for file_data in data:
    models_raw.append([line.strip().split(" ")[1:][0] for line in file_data if line.strip() not in ("New predictions :", "")])
    predictions_raw.append([line.strip().split(" ")[-1][1:] for line in file_data if line.strip() not in ("New predictions :", "")])

print(len(models_raw[0])/12, len(predictions_raw[0])/12)
print(len(models_raw[1])/12, len(predictions_raw[1])/12)
print(len(models_raw[2])/12, len(predictions_raw[2])/12)
raise "Stop here to debug the data extraction and counting before plotting"
# Count predictions per model per file
classes = ["background", "chainsaw", "fire", "fireworks", "gunshot"]

count_dict = {}
for file_idx in range(len(filename)):
    count_dict[filename[file_idx]] = {}
    for model, prediction in zip(models_raw[file_idx], predictions_raw[file_idx]):
        if model not in count_dict[filename[file_idx]]:
            count_dict[filename[file_idx]][model] = {cls: 0 for cls in classes}
        count_dict[filename[file_idx]][model][prediction] += 1

# Normalize to percentages
norm_count_dict = {}
for file_idx in range(len(filename)):
    norm_count_dict[filename[file_idx]] = {}
    for model, counts in count_dict[filename[file_idx]].items():
        total = sum(counts.values())
        norm_count_dict[filename[file_idx]][model] = {cls: (count / total) * 100 for cls, count in counts.items()}

# Build new_dict : model_name -> class -> list of values (one per intensity)
new_dict = {}
for file_idx in range(len(filename)):
    for model, counts in norm_count_dict[filename[file_idx]].items():
        model_class = model.split("_")[0].upper()
        freq = "HF" if "HF" in model else "AF"
        new_name = f"{model_class} - {freq}"
        if new_name not in new_dict:
            new_dict[new_name] = {cls: [] for cls in classes}
        for cls in classes:
            new_dict[new_name][cls].append(counts[cls])

# Plot
model_columns = ["general", "background", "chainsaw", "fire", "fireworks", "gunshot"]
fig, axes = plt.subplots(2, 6, figsize=(24, 10))

x = np.arange(len(classes))          # 5 positions (one per predicted class)
n_bars = len(filename)                # 3 bars per position (one per intensity)
width = 0.25                          # width of each bar
colors_intensity = ["steelblue", "darkorange", "seagreen"]

for model_name, counts in new_dict.items():
    col = model_columns.index(model_name.split(" - ")[0].lower())
    row = 1 if "HF" in model_name else 0
    ax = axes[row, col]

    for i, (intensity_label, color) in enumerate(zip(labels_intensity, colors_intensity)):
        values = [counts[cls][i] for cls in classes]
        ax.bar(x + i * width, values, width, label=intensity_label, color=color)

    ax.set_title(model_name, fontsize=1.5 * fontsize)
    ax.set_ylabel("Count (%)", fontsize=fontsize)
    ax.set_xticks(x + width)  # centre les labels sous les groupes
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=fontsize - 1)
    ax.legend(title="Intensity", fontsize=fontsize - 2)

plt.tight_layout()
plt.savefig(f"predictions_analysis_{filename[0].split('_')[2]}.pdf")

subprocess.Popen(f"predictions_analysis_{filename[0].split('_')[2]}.pdf", shell=True)
