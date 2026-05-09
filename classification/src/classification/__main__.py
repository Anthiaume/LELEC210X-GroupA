# import json
# import pickle
# import threading
# from collections import deque
# from pathlib import Path
# import time

# import click
# import requests

# import common
# from auth import PRINT_PREFIX
# from common.env import load_dotenv
# from common.logging import logger
# from leaderboard.utils import get_url
# import numpy as np

# from .utils import payload_to_melvecs
# from student_fct import *
# load_dotenv()

# # ── Shared state between the packet thread and the submission thread ───────────
# data_queue     = deque()          # filled by the for-loop, consumed by the worker
# queue_lock     = threading.Lock()

# CLASSES        = ["background", "chainsaw", "fire", "fireworks", "gunshot"]
# STOP_SUBMIT    = 4
# RESET_TIME     = 5


# def submit(url, api_key, prediction, name=""):
#     if prediction == "background":
#         print(f"Not submitting {name}'s guess: 'background'.")
#         return None
#     response = requests.post(
#         f"{url}/lelec210x/leaderboard/submit/{api_key}/{prediction}",
#         timeout=1
#     )
#     return json.loads(response.text)


# def majority_vote(predictions):
#     return max(set(predictions), key=predictions.count)


# def majority_vote_arrays(array_list):
#     all_indices = [int(idx) for arr in array_list for idx in arr]
#     best_idx = np.bincount(all_indices).argmax()
#     return CLASSES[best_idx]


# def reset_window():
#     return {
#         "start":                 None,
#         "aggregated_sent":       False,
#         "predictions_spartacus": [],
#         "predictions_joseph":    [],
#         "predictions_gisele":    [],
#         "all_raw_predictions":   [],
#     }


# def submission_worker(url, key, jacqueline_key, jean_key, super_model):
#     """
#     Runs in a background thread.
#     Pulls melvecs from data_queue, predicts, and handles the time window logic.
#     """
#     state = reset_window()
#     set_bool_false()

#     while True:
#         now     = time.time()
#         elapsed = (now - state["start"]) if state["start"] else 0

#         # ── [STOP_SUBMIT, RESET_TIME): send aggregated votes once, then idle ──
#         if state["start"] is not None and elapsed >= STOP_SUBMIT:

#             if not state["aggregated_sent"]:
#                 all_preds_flat = (state["predictions_spartacus"]
#                                 + state["predictions_joseph"]
#                                 + state["predictions_gisele"])
#                 if all_preds_flat:
#                     result = submit(url, jacqueline_key, majority_vote(all_preds_flat), "Jacqueline")
#                     if result:
#                         print("Jacqueline →", result)

#                 if state["all_raw_predictions"]:
#                     result = submit(url, jean_key, majority_vote_arrays(state["all_raw_predictions"]), "Jean")
#                     if result:
#                         print("Jean →", result)

#                 state["aggregated_sent"] = True

#             if elapsed >= RESET_TIME:
#                 state = reset_window()
#                 set_bool_false()

#             time.sleep(0.05)
#             continue

#         # ── [0, STOP_SUBMIT): consume queued melvecs ──────────────────────────
#         with queue_lock:
#             melvecs = data_queue.popleft() if data_queue else None

#         if melvecs is None:
#             time.sleep(0.01)
#             continue

#         # Start window on first received data
#         if state["start"] is None:
#             state["start"] = time.time()

#         # --- Predict ---
#         models    = load_models()
#         raw_preds = np.zeros(len(models), dtype=int)
#         for i, m in enumerate(models):
#             x         = process_data_for_MLP(melvecs, m["params"])
#             raw_preds[i] = int(m["model"].predict(x)[0])

#         spartacus_pred = CLASSES[np.bincount(raw_preds).argmax()]
#         joseph_pred    = CLASSES[super_model.predict(raw_preds.reshape(1, -1))[0]]
#         string_preds   = [CLASSES[i] for i in raw_preds]
#         gisele_pred, _ = rule_grand_mere(string_preds)

#         state["predictions_spartacus"].append(spartacus_pred)
#         state["predictions_joseph"].append(joseph_pred)
#         state["predictions_gisele"].append(gisele_pred)
#         state["all_raw_predictions"].append(raw_preds)

#         for api_key, pred, name in [
#             (key,          spartacus_pred, "Spartacus"),
#             (jacqueline_key, joseph_pred,  "Joseph"),   # adapte les clés selon ton setup
#             (jean_key,     gisele_pred,    "Gisele"),
#         ]:
#             result = submit(url, api_key, pred, name)
#             if result:
#                 print(f"{name} →", result)


# # ── Click command ──────────────────────────────────────────────────────────────

# @click.command()
# @click.option("-i", "--input", "_input", default="-", type=click.File("r"))
# @click.option("-m", "--model", default=None, type=click.Path(exists=True, dir_okay=False, path_type=Path))
# @common.click.melvec_length
# @common.click.n_melvecs
# @click.option("--submit/--no-submit", default=True)
# @click.option("-u", "--url", default=None, envvar="LEADERBOARD_URL")
# @click.option("-k", "--key", default=None, envvar="LEADERBOARD_KEY")
# @click.option("--jacqueline-key", default=None, envvar="JACQUELINE_KEY")
# @click.option("--jean-key", default=None, envvar="JEAN_KEY")
# @common.click.verbosity
# def main(_input, model, melvec_length, n_melvecs, submit, url, key,
#          jacqueline_key, jean_key):

#     if submit:
#         if key is None:
#             raise click.UsageError("You must provide a key to submit guesses.")
#         url = url or get_url()

#     super_model = load_super_model()

#     # Start the submission worker in a background thread
#     worker = threading.Thread(
#         target=submission_worker,
#         args=(url, key, jacqueline_key, jean_key, super_model),
#         daemon=True   # killed automatically when main exits
#     )
#     worker.start()

#     # Main thread: just parse packets and push melvecs into the queue
#     for payload in _input:
#         if PRINT_PREFIX in payload:
#             payload = payload[len(PRINT_PREFIX):]
#             melvecs = payload_to_melvecs(payload, melvec_length, n_melvecs)
#             logger.info(f"Parsed payload into Mel vectors: {melvecs}")

#             with queue_lock:
#                 data_queue.append(melvecs)

import json
import pickle
from pathlib import Path
import time

import click
import requests

import common
from auth import PRINT_PREFIX
from common.env import load_dotenv
from common.logging import logger
from leaderboard.utils import get_url
import numpy as np

from .utils import payload_to_melvecs
from student_fct import *
load_dotenv()


@click.command()
@click.option(
    "-i",
    "--input",
    "_input",
    default="-",
    type=click.File("r"),
    help="Where to read the input stream. Default to '-', a.k.a. stdin.",
)
@click.option(
    "-m",
    "--model",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the trained classification model.",
)
@common.click.melvec_length
@common.click.n_melvecs
@click.option(
    "--submit/--no-submit",
    default=True,
    help="Whether to submit the guesses to the leaderboard.",
)
@click.option(
    "-u",
    "--url",
    default=None,
    envvar="LEADERBOARD_URL",
    show_default=True,
    show_envvar=True,
    help="Base API url. If not specified, will use FLASK_RUN_HOST and FLASK_RUN_PORT.",
)
@click.option(
    "-k",
    "--key",
    default=None,
    envvar="LEADERBOARD_KEY",
    show_envvar=True,
    help="Your private key.",
)
@common.click.verbosity
def main(
    _input: click.File | None,
    model: Path | None,
    melvec_length: int,
    n_melvecs: int,
    submit: bool,
    url: str | None,
    key: str | None,
) -> None:
    """
    Extract Mel vectors from payloads and perform classification on them.
    Classify MELVECs contained in payloads (from packets).

    Most likely, you want to pipe this script after running authentification
    on the packets:

        uv run auth | uv run classify

    This way, you will directly receive the authentified packets from STDIN
    (standard input, i.e., the terminal).
    """

    global begin_time, models, predicted_probabilities, predicted_classes, pca_models
    print("Jpeux pas décrocher, ya NOUNOURS")
    if submit:
        if key is None:
            raise click.UsageError("You must provide a key to submit guesses.")
        url = url or get_url()
    #if model:
        # with open(model, "rb") as file:
        #     m = pickle.load(file)
#else:
    #    m = None

    for payload in _input:
        if PRINT_PREFIX in payload:
            payload = payload[len(PRINT_PREFIX) :]

            melvecs = payload_to_melvecs(payload, melvec_length, n_melvecs)
            logger.info(f"Parsed payload into Mel vectors: {melvecs}")

            ###########################################################
            ### BEGIN STUDENT MODIFICATIONS ###########################
            ###########################################################
            save = False
            if save:
                pickle.dump(melvecs, open(f"Melvec_{time.ctime()}.pkl", "wb"))
                print(f"Melvecs saved in file Melvec_{time.ctime()}.pkl")

            models = load_models()
            predictions = np.zeros(len(models))
            for current_model in range(len(models)):
                x_test_processed = process_data_for_MLP(melvecs, models[current_model]["params"])
                y_pred = models[current_model]["model"].predict(x_test_processed)
                predictions[current_model] = y_pred[0]
            prediction = int(np.bincount(predictions.astype(int)).argmax())
            classes = ["background", "chainsaw", "fire", "fireworks", "gunshot"]
            guess = classes[prediction]

            print("GUESS = |", guess, "|", time.ctime(), sep="")
            print(type(guess))
            # Submit the guess to the leaderboard if required
            if submit and guess != "background" and not get_bool_submitt_good():
                response = requests.post(
                    f"{url}/lelec210x/leaderboard/submit/{key}/{guess}"
                )

                response_as_dict = json.loads(response.text)

                if response.status_code == 200:
                    logger.info(response_as_dict)
                    bool_response = response_as_dict['penalized']
                    if bool_response:
                        change_bool_submitt_good()
                else:
                    logger.error(response_as_dict)
            ###########################################################
            ### END STUDENT MODIFICATIONS #############################
            ###########################################################