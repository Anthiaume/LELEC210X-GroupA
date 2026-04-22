import pickle


rounds = ["round 1.txt", "round 2.txt", "round 4.txt"]

interesting_data = []
for round in rounds:
    with open(round, "r") as f:
        output = f.readlines()
        output = [line.strip().split("\t") for line in output]
        interesting_data.extend(output)

# # list of files in the directory
from os import listdir
from os.path import isfile, join
import os

index = 1
original_files = [file for file in os.listdir() if file.endswith(".pkl") and file.startswith("Melvec")]
files = [file.replace("\uf03a", ":").removeprefix("Melvec_").removesuffix(".pkl") for file in os.listdir() if file.endswith(".pkl") and file.startswith("Melvec")]
useless_files = original_files.copy()

for file in range(len(files)):
    for fv in range(len(interesting_data)):
        if files[file] == interesting_data[fv][0]:

            with open(original_files[file], "rb") as f:
                data = pickle.load(f)

            TYPE = interesting_data[fv][2]
            if TYPE in ["fire", "chainsaw", "gunshot", "fireworks"]:
                mypath = join(os.getcwd(), "labeled")
                onlyfiles = [f  for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith(".pkl") and f.startswith(TYPE))]
                maximum = max([int(f.split('_')[1].strip('.pkl')) for f in onlyfiles if f.startswith(TYPE) and f.endswith(".pkl")] , default=0)
                pickle.dump(data, open(f"labeled/{TYPE}_{maximum+1}.pkl", "wb"))
                # print(f"{index} - {TYPE}_{maximum+1}.pkl")
                useless_files.remove(original_files[file])

for file in useless_files:
    TYPE = "voices"
    with open(file, "rb") as f:
        data = pickle.load(f)
    onlyfiles = [f  for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith(".pkl") and f.startswith(TYPE))]
    maximum = max([int(f.split('_')[1].strip('.pkl')) for f in onlyfiles if f.startswith(TYPE) and f.endswith(".pkl")] , default=0)
    pickle.dump(data, open(f"labeled/{TYPE}_{maximum+1}.pkl", "wb"))
    
            


# with open("output computer.txt", "r") as f:
#     output = f.readlines()

# n_label = 0
# labels = []
# data = []
# for line in range(len(output)):
#     if output[line].startswith("**GUESS"):
#         label = output[line].split("|")[1]
#         datum  = output[line].split("|")[2][:-3]
#         labels.append(label)
#         data.append(datum)

# with open("synthesis.txt", "w") as f:
#     for i in range(len(labels)):
#         f.write(f"{data[i]}\t{labels[i]}\n")

# print(set(labels))
# print(len(set(labels)))

# # number of samples per class
# from collections import Counter
# counter = Counter(labels)

# print(counter)

# # list of files in the directory
# import os
# files = os.listdir()
# with open("caracter.txt", "w", encoding="utf-8") as f:
#     for file in files:
#         if file.endswith(".pkl"):
#             string = file.split()[3]
#             string = string[0:2] + "h" +string[3:5] + "m" + string[6:] + "s"
#             if int(string[3:5]) == 42 and int(string[6:8]) >=28:
#                 print(string)
#                 f.write(f"{string}\n")
#             if int(string[3:5]) == 43 or int(string[3:5]) == 44:
#                 print(string)
#                 f.write(f"{string}\n")
#             if int(string[3:5]) == 45 and int(string[6:8]) <= 5:
#                 print(string)
#                 f.write(f"{string}\n")