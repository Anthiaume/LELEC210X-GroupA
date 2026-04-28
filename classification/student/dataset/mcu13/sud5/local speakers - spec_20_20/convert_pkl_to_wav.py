import pickle
import numpy as np
from scipy.io import wavfile
from pathlib import Path
import os

# Paramètres
SAMPLE_RATE = 10205  # À adapter si besoin
OUTPUT_DIR = "converted_wav"

# Crée le dossier de sortie s'il n'existe pas
dir_path = Path(__file__).parent
output_path = dir_path / OUTPUT_DIR
output_path.mkdir(exist_ok=True)

# Parcourt tous les fichiers .pkl du dossier
for pkl_file in dir_path.glob("*.pkl"):
    try:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            # Conversion selon le type
            if np.issubdtype(data.dtype, np.floating):
                data_to_save = (data * 32767).astype(np.int16)
            elif data.dtype == np.uint16:
                # Décale et convertit uint16 (0-65535) en int16 (-32768 à 32767)
                data_to_save = (data.astype(np.int32) - 32768).astype(np.int16)
            elif data.dtype == np.int16:
                data_to_save = data
            else:
                raise ValueError(f"Type de données non supporté: {data.dtype}")
        wav_name = pkl_file.stem + ".wav"
        wav_path = output_path / wav_name
        wavfile.write(wav_path, SAMPLE_RATE, data_to_save)
        print(f"Converti {pkl_file.name} -> {wav_path}")
    except Exception as e:
        print(f"Erreur pour {pkl_file.name}: {e}")

print("Conversion terminée.")
