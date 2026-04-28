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