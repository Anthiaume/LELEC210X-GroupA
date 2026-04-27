from student_fct import *
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


models = load_models()

x_test, y_test = pickle.load(open(f"x_train.pkl", "rb"))

predictions = np.zeros((len(models), len(y_test)))

for model in range(len(models)):
    
    x_test_processed = process_data_for_MLP(x_test, models[model]["params"])

    y_pred = models[model]["model"].predict(x_test_processed)

    predictions[model] = y_pred

# predict the final class by majority voting
final_predictions = []
for i in range(len(y_test)):
    # get the predictions for the current sample
    sample_predictions = predictions[:, i]
    
    # get the most common prediction
    final_prediction = np.bincount(sample_predictions.astype(int)).argmax()
    
    final_predictions.append(final_prediction)
# calculate the accuracy of the final predictions
accuracy = accuracy_score(y_test, final_predictions)
print(f"Final Accuracy: {accuracy}")