# NLPtutorial 02にて作成

import sys  # i/o
import math
from collections import defaultdict  # default-set

model_file = open("model-file2.txt", "r").readlines()
trg_file = open(sys.argv[1], "r").readlines()
prob = defaultdict(lambda: 0)
lambda_1 = 0.95
lambda_2 = 0.95
V = 1000000
W = 0
H = 0
lambda_list = list()

start_value = 0.05
while start_value < 1:
    lambda_list.append(start_value)
    start_value += 0.05

# load model
for line in model_file:
    line_list = line.strip()
    pair = line_list.split(" ")
    key = " ".join(pair[0:len(pair)-1])
    prob[key] = pair[-1]

print(f"lambda_1, lambda_2, entropy")

# calc-lambda-search
for lambda_1 in lambda_list:
    for lambda_2 in lambda_list:
        W = 0
        H = 0
        for line in trg_file:
            words = line.strip().split(" ")
            words.append("</s>")
            words.insert(0, "<s>")
            for i in range(1, len(words)):
                unigram = words[i]
                # bigram = " ".join(words[i:i+2])
                bigram = " ".join(words[i-1:i+1])
                P1 = lambda_1 * float(prob[unigram]) + (1-lambda_1) / V
                P2 = lambda_2 * float(prob[bigram]) + (1-lambda_2) * P1
                H += - math.log2(P2)
                W += 1
        print(f"{lambda_1:.2f}, {lambda_2:.2f}, {+float(+H)/W}")
