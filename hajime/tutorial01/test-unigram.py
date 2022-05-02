import sys  #i/o
import math
from collections import defaultdict #default-set

model_file = open("model-file.txt","r").readlines()
trg_file = open(sys.argv[1],"r").readlines()
plob = dict()
lambda_1 = 0.95
lambda_unk = 1- lambda_1
V = 1000000
W = 0
H = 0
unk = 0

#read model

for line in model_file:
    line_list = line.strip()
    pair = line_list.split(" ")
    plob[pair[0]] = pair[1]

#evaluation and result

for line in trg_file:
    words = line.strip().split(" ")
    words.append("</s>")
    for word in words:
        W += 1
        P = float(lambda_unk / V)
        if word in plob:
            P += lambda_1 * float(plob[word])
        else:
            unk += 1
        H += -math.log(P,2)
print("entropy =",float(+H/W))
print("coverage =",float((W-unk)/W))

