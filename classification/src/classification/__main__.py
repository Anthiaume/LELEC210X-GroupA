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

load_dotenv()

global begin_time, models, predicted_probabilities, predicted_classes, pca_models
begin_time = 0 # Equivaut au 1 janvier 1970, 00:00:00 UTC, soit une date très ancienne, pour forcer l'initialisation des variables lors du premier appel à student_var_initialization().

def student_var_initialization():
    global begin_time, models, predicted_probabilities, predicted_classes, pca_models
    # Initialization of the begin time
    begin_time = time.time()

    # Load the models
    models = ["MLP_bithynion.pkl"]#, "KNN_albaniana.pkl"]
    for model in range(len(models)):
        with open(models[model], "rb") as f:
            models[model] = pickle.load(f)

    # Load the PCA models
    # pca_models = ["PCA_albaniana.pkl"]
    # for i, model in enumerate(pca_models):
    #     with open(model, "rb") as f:
    #         pca_models[i] = pickle.load(f)

    # Initialization of the prediction probabilities
    predicted_probabilities = [ [] for i in range(len(models))]
    # Initialization of the predicted classes
    predicted_classes = [ [] for i in range(len(models))]

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
            melvecs_normalized = melvecs / np.linalg.norm(melvecs.flatten(), axis=0, keepdims=True)

            if time.time() - begin_time > 6:
                student_var_initialization()

            for model in range(len(models)):
                predicted_classes[model].append(models[model].predict(melvecs_normalized.reshape(1, -1))[0])
                predicted_probabilities[model].append(models[model].predict_proba(melvecs_normalized.reshape(1, -1))) 

            predictions = []
            for model in range(len(models)):
                sum = np.zeros(predicted_probabilities[model][0].shape[1])
                for i in range(len(predicted_probabilities[model])):
                    sum += predicted_probabilities[model][i].reshape(5)
                predictions.append(models[model].classes_[np.argmax(sum)])
            guess = str(max(set(predictions), key=predictions.count))

            # correction crackling fire -> fire
            if guess == "crackling fire":
                guess = "fire"
            print("GUESS = |", guess, "|", sep="")
            print(type(guess))
            # Submit the guess to the leaderboard if required
            if submit and guess != "background":
                response = requests.post(
                    f"{url}/lelec210x/leaderboard/submit/{key}/{guess}"
                )

                response_as_dict = json.loads(response.text)

                if response.status_code == 200:
                    logger.info(response_as_dict)
                else:
                    logger.error(response_as_dict)
            ###########################################################
            ### END STUDENT MODIFICATIONS #############################
            ###########################################################