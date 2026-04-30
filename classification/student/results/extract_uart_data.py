import matplotlib.pyplot as plt
import subprocess
import warnings

warnings.filterwarnings("ignore")

# Define the filename and the real class for analysis
filesnames= ["results_uart_gunshot_50.txt", "results_uart_gunshot_100.txt"]
filename = "results_uart_gunshot_25.txt"
real_class = "gunshot" # Change this to the real class you want to analyze
fontsize = 12

# Read the data from the file
with open(filename, 'r') as f:
    lines = f.readlines()

# Extract model names and predictions from the lines
models      = [line.strip().split(" ")[1:][0] for line in lines if (line.strip() != "New predictions :" and line.strip() != "")]
predictions = [line.strip().split(" ")[-1][1:] for line in lines if (line.strip() != "New predictions :" and line.strip() != "")]

# Count the predictions for each model
count_dict = {}
for model, prediction in zip(models, predictions):
    if model not in count_dict:
        count_dict[model] = {
            "background": 0,
            "chainsaw": 0,
            "fire": 0,
            "fireworks": 0,
            "gunshot": 0
        }
    count_dict[model][prediction] += 1

# Normalize the counts to get percentages
norm_count_dict = {}
for model, counts in count_dict.items():
    total = sum(counts.values())
    norm_count_dict[model] = {cls: (count / total) * 100 for cls, count in counts.items()}

# Rename the models for better visualization
norm_count_dict_renamed = {}
for model in sorted(norm_count_dict.keys()):
    model_class = model.split("_")[0].upper()
    freq_model = "HF" if "HF" in model else "AF"
    new_name = f"{model_class} - {freq_model}"
    norm_count_dict_renamed[new_name] = norm_count_dict.pop(model)

# Create subplots for each model
fig, axes = plt.subplots(2, 6, figsize=(20, 10))
models = ["general", "background", "chainsaw", "fire", "fireworks", "gunshot"]
indexes = [0, 1, 2, 3, 4, 5]
colors = ["blue", "orange", "green", "red", "purple", "brown"]
for i, (model, counts) in enumerate(norm_count_dict_renamed.items()):
    column = models.index(model.split(" - ")[0].lower())
    row = 1 if "HF" in model else 0
    color = colors[column]
    axes[row, column].bar(counts.keys(), counts.values(), color=color)
    axes[row, column].set_title(model, fontsize=1.75*fontsize)
    # axes[row, column].set_xlabel("Predicted Class", fontsize=1.5*fontsize)
    axes[row, column].set_ylabel("Count (%)", fontsize=1.5*fontsize)
    axes[row, column].set_xticklabels(counts.keys(), rotation=45, fontsize=fontsize)
    axes[row, column].set_yticklabels(axes[row, column].get_yticklabels(), fontsize=fontsize)
plt.tight_layout()
plt.savefig(f"predictions_analysis_{filename.split('_uart_')[1].split('.')[0]}.pdf")

# Open the PDF file using the default PDF viewer
subprocess.Popen(f"predictions_analysis_{filename.split('_uart_')[1].split('.')[0]}.pdf", shell=True)