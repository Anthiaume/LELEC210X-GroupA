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

for i in range(len(models)):
    print(f"{i+1}. {models[i]['model_name']}")

x_test, y_test = pickle.load(open(f"x_test.pkl", "rb"))
classes = ["general", "background", "chainsaw", "fireworks", "fire", "gunshot"]

# Display Normalized Confusion Matrices
n_plots = len(models)
square = np.arange(1,10)**2
n_axes = int(n_plots**0.5) if n_plots in square else int(np.ceil(np.sqrt(n_plots)))
factor = 1
fig, axes = plt.subplots(4, 3, figsize=(4*n_axes*factor, 6*n_axes*factor))
axes = axes.flatten() if n_axes > 1 else [axes]
for model in range(len(models)):
    x_test_processed = process_data_for_MLP_for_test(x_test, models[model]["params"])

    y_pred = models[model]["model"].predict(x_test_processed)

    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred).astype(float)

    print(f"{models[model]['model_name']} Accuracy: {accuracy}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm/np.sum(cm)*100, display_labels=models[model]["classes"])
    disp.plot(cmap=plt.cm.Blues, ax=axes[model], colorbar=False)
    disp.ax_.set_xticklabels(models[model]['classes'], rotation=45, ha='right', rotation_mode='anchor', fontsize=fontsize)
    disp.ax_.set_yticklabels(models[model]['classes'], fontsize=fontsize)
    disp.ax_.set_xlabel("Predicted Label", fontsize=fontsize*1.2)
    disp.ax_.set_ylabel("True Label", fontsize=fontsize*1.2)
    title = ""
    for cls in classes:
        if cls in models[model]["model_name"].lower():
            title = cls.capitalize()
            break
    title += " " + "HF" if "hf" in models[model]["model_name"].lower() else " AF"
    title += f"\nAccuracy: {accuracy:.1%}"
    axes[model].set_title(title, fontsize=fontsize*1.5)

for j in range(model+1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
if 1:
    plt.savefig(f"CONFUSION_MATRICES_COMPARISON_final_models.pdf", bbox_inches='tight')
if 1:
    plt.savefig(f"temp.pdf", bbox_inches='tight')
    subprocess.Popen(["temp.pdf"],shell=True)
    time.sleep(1)
    os.remove(f"temp.pdf")
plt.close()