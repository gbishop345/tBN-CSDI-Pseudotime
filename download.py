import tarfile
import sys
import os
import wget

os.makedirs("data/", exist_ok=True)
if sys.argv[1] == "physio":
    url = "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download"
    wget.download(url, out="data")
    with tarfile.open("data/set-a.tar.gz", "r:gz") as t:
        t.extractall(path="data/physio")

