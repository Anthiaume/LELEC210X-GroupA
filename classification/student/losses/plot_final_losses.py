from student_fct import *
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

fontsize = 14

models = load_models()

models.sort(key=lambda x: x["model_name"].split("_")[0]+x["model_name"].split("_")[2])

models_general = models[8:10]
models_gunshot = models[10:]
models[8:10] = models_gunshot
models[10:] = models_general

names = []
for i in range(len(models)):
    # print(f"{i+1}. {models[i]['model_name']}")
    name = models[i]["model_name"].split("_")[0].upper()
    name += " - HF " if "hf" in models[i]["model_name"].lower() else " - AF"

    names.append(name)
#######################################################################################

files_in_folder = os.listdir()
files_in_folder = [f for f in files_in_folder if f.endswith(".pkl")]
data = {}
for f in range(len(models)):
    data[names[f]] = pickle.load(open(f"train_val_losses_{models[f]['model_name']}.pkl", "rb"))

#######################################################################################

classes = ["general", "background", "chainsaw", "fireworks", "fire", "gunshot"]

# Display Normalized Confusion Matrices
n_plots = len(models)
square = np.arange(1,10)**2
n_axes = int(n_plots**0.5) if n_plots in square else int(np.ceil(np.sqrt(n_plots)))
factor = 1
fig, axes = plt.subplots(4, 3, figsize=(4*n_axes*factor, 5*n_axes*factor))
axes = axes.flatten() if n_axes > 1 else [axes]
for model in range(len(models)):
    axes[model].plot(data[names[model]][0], label="Train losses")
    axes[model].plot(data[names[model]][1], label="Validation losses")
    axes[model].set_title(list(data.keys())[model], fontsize=fontsize*1.5)
    axes[model].legend(fontsize=fontsize*1.4)


for j in range(model+1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
if 1:
    plt.savefig(f"LOSSES_final_models.pdf", bbox_inches='tight')
if 1:
    plt.savefig(f"temp.pdf", bbox_inches='tight')
    subprocess.Popen(["temp.pdf"],shell=True)
    time.sleep(1)
    os.remove(f"temp.pdf")
plt.close()