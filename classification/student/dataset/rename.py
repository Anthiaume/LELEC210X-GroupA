import os
from os import listdir
from os.path import isfile, join

OLDTYPE = "chainsaw" # "chainsaw", "gunshot", "fireworks", "crackling fire", "background"
NEWTYPE = "chainsaw"
mypath = os.getcwd()#os.path.join(SPEAKER)

onlyfiles = [f  for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith(".pkl") and f.startswith(OLDTYPE))]
# sort onlyfiles by number at the end of the filename
onlyfiles.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
print(onlyfiles)

for f in range(len(onlyfiles)):
    os.rename(onlyfiles[f], f"{NEWTYPE}_{f+1}.pkl")