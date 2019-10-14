
import urllib.request
import pandas as pd
from itertools import product

# Generate combinations and their respective lists
amino_acids = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "O", "I", "L", "K",
               "M", "F", "P", "U", "S", "T", "W", "Y", "V"]

combos = list(product(amino_acids, repeat=2))
running_epitopes = []

for i in combos:
    print("".join(i))
    with urllib.request.urlopen("http://www.ddg-pharmfac.net/antijen/scripts/"
                                + "aj_scripts/aj_tapcalc.pl?epitope="
                                + "".join(i)
                                + "&MIN=&MAX=&allele=&CATEGORY=TAP&ic50MIN="
                                + "&ic50MAX=&KDNMMIN=&KDNMMAX=&TAP=Search+AntiJen") as h:
        epitope_buffer = str(h.read()).split("epitope value=")
        for j in epitope_buffer[1:]:
            running_epitopes.append(j.split(">")[0])

running_epitopes = list(dict.fromkeys(running_epitopes))

pd.DataFrame({"Description": running_epitopes}).to_csv("/Users/weeder/PycharmProjects/proteasome/data_extraction/raw_data/AntiJen/TAP.csv", index=False)
