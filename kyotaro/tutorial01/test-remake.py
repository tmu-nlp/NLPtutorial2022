import sys
import math
from collections import defaultdict


train_result = open(sys.argv[1], "r").readlines()
test_file = open(sys.argv[2], "r").readlines()

w = 0
lmd = 0.95
V = 1000000
unk = 0
h = 0
probabilities = defaultdict(lambda: 0)

for line in train_result:
    line = line.strip()
    words = line.split()
    probabilities[words[0]] = words[1]


for line in test_file:
    line = line.strip()
    words = line.split()
    words.append("</s>")
    for i in words:
        w += 1
        p = (1 - lmd) / V
        if probabilities[i]:
            p += lmd * float(probabilities[i])
        else:
            unk += 1
        h += (-1) * math.log(p, 2)

print("entropy = ", h / w)
print("coverage = ", (w - unk) / w)