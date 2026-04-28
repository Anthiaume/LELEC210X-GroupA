import pickle
import os

files = [file for file in os.listdir() if file.endswith(".pkl")]

for file in files:
    with open(file, "rb") as f:
        data = pickle.load(f)
    data = data.reshape(1, -1).flatten()
    pickle.dump(data, open(file, "wb"))
    print(f"Reshaped and saved {file} with shape {data.shape}")