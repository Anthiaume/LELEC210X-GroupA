import random
from pathlib import Path

import click
import numpy as np
from pydub import AudioSegment
from tqdm import trange

import common

from ..datasets import SOUND_DURATION


def get_shot_offset_ms(audio: AudioSegment) -> float:
    """Return time (ms) of strongest impulse in full clip."""
    samples = np.array(audio.get_array_of_samples())
    peak_index = np.argmax(np.abs(samples))
    return (peak_index / audio.frame_rate) * 1000


def random_time_gen(n, start, end, min_dist):
    """
    Generate n floats in [start, end] with at least min_dist between them.
    """
    total_range = end - start
    required_space = min_dist * (n - 1)

    slack = total_range - required_space

    if slack < 0:
        raise ValueError("Range too small for the given n and min_dist")

    random_gaps = np.random.rand(n + 1)
    scaled_gaps = random_gaps / random_gaps.sum() * slack
    scaled_gaps[1:] += min_dist
    return start + np.cumsum(scaled_gaps)


@click.command()
@click.argument(
    "sources",
    nargs=-1,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-n",
    "--num_clips",
    default=10,
    show_default=True,
    type=click.IntRange(min=1),
    help="How many audio clips to produce.",
)
@click.option(
    "-d",
    "--duration",
    default=SOUND_DURATION,
    show_default=True,
    type=click.FloatRange(min=0.0, min_open=True),
    help="How long each clip should be. (s)",
)
@click.option(
    "-s",
    "--seed",
    default=None,
    type=click.IntRange(min=0),
    help="Random seed to use.",
)
@click.option(
    "--directory",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("."),
    show_default=True,
    help="Output directory for generated audio files.",
)
@click.option(
    "-p",
    "--prefix",
    type=str,
    default="gunshots",
    help="Filename prefix for generated audio files. "
    "If not specified, uses SOURCE's filename.",
)
@click.option(
    "-o",
    "--occurences",
    default=3,
    show_default=True,
    type=click.IntRange(min=1),
    help="How occurences of the sound to play.",
)
@click.option(
    "-t",
    "--time_delta",
    default=1,
    show_default=True,
    type=click.IntRange(min=0),
    help="Minimal time between sounds. (s)",
)
@click.option(
    "--slack",
    default=500,
    show_default=True,
    type=click.IntRange(min=0),
    help="Gain to reduce the background intensity. (dB)",
)

@click.option(
    "--background",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Background audio to play during the whole clip.",
)
@click.option(
    "--bg_gain",
    default=-15,
    show_default=True,
    type=click.IntRange(min=-60, max=0),
    help="Gain applied to the background audio (dB).",
)

@common.click.verbosity
def main(
    sources: tuple[Path, ...],
    num_clips: int,
    occurences: int,
    duration: float,
    time_delta: float,
    slack: float,
    seed: int | None,
    directory: Path,
    prefix: str | None,
    background: tuple[Path, ...],
    bg_gain: int | None,
) -> None:
    random.seed(seed)
    directory.mkdir(parents=True, exist_ok=True)

    # Si un seul argument est donné et c'est un dossier, prendre tous les .wav dedans
    if len(sources) == 1 and sources[0].is_dir():
        folder = sources[0]
        sources = tuple(sorted(folder.glob("*.wav")))
        if not sources:
            raise ValueError(f"Aucun fichier .wav trouvé dans le dossier {folder}")

    # Vérifier qu'il y a au moins un fichier source
    if not sources:
        raise ValueError("Aucun fichier source fourni.")

    duration_ms = duration * 1000
    delta_ms = time_delta * 1000

    cache: dict[Path, tuple[AudioSegment, float]] = {}

    for clip_index in trange(num_clips, desc="Generating audio files..."):
        if background:
            bg = AudioSegment.from_wav(background).apply_gain(bg_gain)

            # boucler le background pour remplir toute la durée
            repeats = int(duration_ms / len(bg)) + 1
            piece = (bg * repeats)[:int(duration_ms)]
        else:
            piece = AudioSegment.silent(duration=int(duration_ms))

        max_start = duration_ms - (occurences - 1) * delta_ms
        if max_start < 0:
            raise ValueError(
                "Duration too short for required occurrences and time_delta."
            )

        current_time = slack

        # choisir les sons dans un ordre aléatoire
        chosen_sounds = random.sample(list(sources), k=occurences) if len(sources) >= occurences else random.choices(sources, k=occurences)

        for path in chosen_sounds:

            if path not in cache:
                audio = AudioSegment.from_wav(path)
                audio = audio.apply_gain(-audio.max_dBFS)
                cache[path] = audio

            audio = cache[path]

            # placer le son
            piece = piece.overlay(audio, position=int(current_time))

            # avancer le temps : durée du son + espace minimum
            current_time += len(audio) + delta_ms

            if current_time > duration_ms:
                raise ValueError(
                    "Clip trop court pour placer tous les sons avec l'intervalle demandé."
                )

        filename = f"{prefix or 'clip'}_{clip_index:02d}.wav"
        piece.export(directory / filename, format="wav")
