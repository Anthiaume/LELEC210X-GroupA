import requests
import time
import json
import numpy as np
from student_fct import *

hostname       = "http://localhost:5000"
key            = "beyUzBXz05-tpxRQoGIvUKk-SbM3BSa-zrVp_MKa"
gisele_key     = "isPK_jZBdZ8n9-eGQP1lL-laOIwLo21trvCRN3Gw"
joseph_key     = "K_WC8bjtQNCetT0eSzrIfi6GFjZLIdwoEThg0s-I"
jacqueline_key = "8meaS24u7f53sef9s6J11twG5HOyma5qk24k4xDY"
jean_key       = "G2dH1goy-z6hkkKG5Glh8VMFTl3saPweI9E9gCXi"

CLASSES        = ["background", "chainsaw", "fire", "fireworks", "gunshot"]
STOP_SUBMIT    = 4   # seconds: stop collecting, send aggregated votes
RESET_TIME     = 7   # seconds: reset everything


def submit(api_key, prediction, name=""):
    if prediction == "background":
        print(f"Not submitting {name}'s guess: 'background'.")
        return None
    response = requests.post(
        f"{hostname}/lelec210x/leaderboard/submit/{api_key}/{prediction}",
        timeout=1
    )
    return json.loads(response.text)


def majority_vote(predictions):
    return max(set(predictions), key=predictions.count)


def majority_vote_arrays(array_list):
    all_indices = [int(idx) for arr in array_list for idx in arr]
    best_idx = np.bincount(all_indices).argmax()
    return CLASSES[best_idx]


def reset_window():
    return {
        "start":               None,
        "aggregated_sent":     False,
        "predictions_spartacus": [],
        "predictions_joseph":    [],
        "predictions_gisele":    [],
        "all_raw_predictions":   [],
    }


# --- Initialization ---
super_model = load_super_model()
set_bool_false()
state = reset_window()

# --- Main loop ---
while True:
    now = time.time()

    # ── Phase detection ────────────────────────────────────────────────────────
    if state["start"] is None:
        elapsed = 0
    else:
        elapsed = now - state["start"]

    # ── [STOP_SUBMIT, RESET_TIME): idle + send aggregated votes once ───────────
    if state["start"] is not None and elapsed >= STOP_SUBMIT:

        if not state["aggregated_sent"]:
            all_preds_flat = (state["predictions_spartacus"]
                            + state["predictions_joseph"]
                            + state["predictions_gisele"])
            if all_preds_flat:
                result = submit(jacqueline_key, majority_vote(all_preds_flat), "Jacqueline")
                if result:
                    print("Jacqueline →", result)

            if state["all_raw_predictions"]:
                result = submit(jean_key, majority_vote_arrays(state["all_raw_predictions"]), "Jean")
                if result:
                    print("Jean →", result)

            state["aggregated_sent"] = True

        # Reset after RESET_TIME
        if elapsed >= RESET_TIME:
            state = reset_window()
            set_bool_false()

        time.sleep(0.01)
        continue

    # ── [0, STOP_SUBMIT): collect data ─────────────────────────────────────────
    data = get_data()   # only called when we're allowed to collect

    if data is None:
        time.sleep(0.01)
        continue

    # Start the window on first received data
    if state["start"] is None:
        state["start"] = time.time()

    # --- Predict ---
    models   = load_models()
    raw_preds = np.zeros(len(models), dtype=int)

    for i, m in enumerate(models):
        x = process_data_for_MLP(data, m["params"])
        raw_preds[i] = int(m["model"].predict(x)[0])

    spartacus_pred = CLASSES[np.bincount(raw_preds).argmax()]
    joseph_pred    = CLASSES[super_model.predict(raw_preds.reshape(1, -1))[0]]
    string_preds   = [CLASSES[i] for i in raw_preds]
    gisele_pred, _ = rule_grand_mere(string_preds)

    state["predictions_spartacus"].append(spartacus_pred)
    state["predictions_joseph"].append(joseph_pred)
    state["predictions_gisele"].append(gisele_pred)
    state["all_raw_predictions"].append(raw_preds)

    for api_key, pred, name in [
        (key,        spartacus_pred, "Spartacus"),
        (joseph_key, joseph_pred,    "Joseph"),
        (gisele_key, gisele_pred,    "Gisele"),
    ]:
        result = submit(api_key, pred, name)
        if result:
            print(f"{name} →", result)




# import requests
# import time
# from student_fct import *

# hostname = "http://localhost:5000"
# key = "beyUzBXz05-tpxRQoGIvUKk-SbM3BSa-zrVp_MKa"
# gisele = "isPK_jZBdZ8n9-eGQP1lL-laOIwLo21trvCRN3Gw"
# joseph = "K_WC8bjtQNCetT0eSzrIfi6GFjZLIdwoEThg0s-I"
# jacqueline = "8meaS24u7f53sef9s6J11twG5HOyma5qk24k4xDY"
# jean = "G2dH1goy-z6hkkKG5Glh8VMFTl3saPweI9E9gCXi"

# model = load_super_model()

# timing = []
# predictions_spartacus = []
# predictions_joseph = []
# predictions_gisele = []
# all_predictions_all_models = []
# set_bool_false()

# while True:

        
#     data = get_data()

#     if data is not None:
        
#         timing.append(time.time())

#         if time.time() - timing[0] > 4:
#             timing = []
#             predictions_spartacus = []
#             predictions_joseph = []
#             predictions_gisele = []
#             all_predictions = predictions_spartacus + predictions_joseph + predictions_gisele
#             all_predictions_best = max(set(all_predictions), key=all_predictions.count)
#             if all_predictions_best != "background":
#                 response_jacqueline = requests.post(f"{hostname}/lelec210x/leaderboard/submit/{jacqueline}/{all_predictions_best}", timeout=1)
#             else:
#                 print("Not submitting guess to server since it is 'background'.")
#             all_predictions_all_models_best = max(set(all_predictions_all_models), key=all_predictions_all_models.count)
#             if all_predictions_all_models_best != "background":
#                 response_jean = requests.post(f"{hostname}/lelec210x/leaderboard/submit/{jean}/{all_predictions_all_models_best}", timeout=1)
#             else:
#                 print("Not submitting guess to server since it is 'background'.")
#             all_predictions_all_models = []
#             time.sleep(1)
#             set_bool_false()
#         else:
#             models = load_models()

#             predictions = np.zeros(len(models))

#             for current_model in range(len(models)):
#                 x_test_processed = process_data_for_MLP(data, models[current_model]["params"])
#                 y_pred = models[current_model]["model"].predict(x_test_processed)
#                 predictions[current_model] = y_pred[0]

#             prediction1 = int(np.bincount(predictions.astype(int)).argmax())
#             classes = ["background", "chainsaw", "fire", "fireworks", "gunshot"]
#             guess = classes[prediction1]

#             joseph_pred = classes[model.predict(np.array(predictions).reshape((1,12)))[0]]

#             predictions_pour_gisele = [classes[int(pred)] for pred in predictions]
#             gisele_pred, _ = rule_grand_mere(predictions_pour_gisele)

#             predictions_spartacus.append(guess)
#             predictions_joseph.append(joseph_pred)
#             predictions_gisele.append(gisele_pred)
#             all_predictions_all_models.append(predictions)
            
#             if guess != "background":
#                 response1 = requests.post(f"{hostname}/lelec210x/leaderboard/submit/{key}/{guess}", timeout=1)
#             else:
#                 print("Not submitting guess to server since it is 'background'.")
#             if joseph_pred != "background":
#                 response_joseph = requests.post(f"{hostname}/lelec210x/leaderboard/submit/{joseph}/{joseph_pred}", timeout=1)
#             else:
#                 print("Not submitting Joseph's guess to server since it is 'background'.")
#             if gisele_pred != "background":
#                 response_gisele = requests.post(f"{hostname}/lelec210x/leaderboard/submit/{gisele}/{gisele_pred}", timeout=1)
#             else:
#                 print("Not submitting Gisele's guess to server since it is 'background'.")

#             # N.B.: the timeout is generally a good idea to avoid blocking infinitely (if an error occurs)
#             # but you can change its value. Note a too small value may not give the server enough time
#             # to reply.



#             import json

#             # All responses are JSON dictionaries
#             response_as_dict = json.loads(response1.text)
#             response_joseph_as_dict = json.loads(response_joseph.text)
#             response_gisele_as_dict = json.loads(response_gisele.text)

#             print("response from server (Spartacus):", response_as_dict)
#             print("response from server (Joseph):", response_joseph_as_dict)
#             print("response from server (Gisele):", response_gisele_as_dict)

#     time.sleep(0.05)
