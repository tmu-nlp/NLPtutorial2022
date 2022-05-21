import sys
import math
from collections import defaultdict

train = defaultdict(lambda: 0)
V = 1000000
h = 0
w = 0
test = open("wiki-en-test.word", "r").readlines()
out_file = open("wiki-test-result.word", "w")


with open(sys.argv[1], "r") as train_result:
    for line in train_result:
        line = line.strip()
        words = line.split()
        if len(words) >= 3:
            train[words[0] + " " + words[1]] = words[2]
        else:
            train[words[0]] = words[1]


for lmd1 in range(5, 100, 5):
    lmd1 *= 0.01
    for lmd2 in range(5, 100, 5):
        lmd2 *= 0.01
        w = 0
        h = 0
        for line in test:
            line = line.strip()
            words = line.split()
            words.append("</s>")
            words.insert(0, "<s>")
            for i in range(1, len(words)):
                bi = " ".join(words[i-1:i])
                p1 = lmd1 * float(train[words[i]]) + (1 - lmd1) / V
                p2 = lmd2 * float(train[bi]) + (1 - lmd2) * p1
                h += (-1) * math.log(p2, 2)
                w += 1
        out_file.write("lambda_1 : " + str('{:.2f}'.format(lmd1)) + " lambda_2 :" + str('{:.2f}'.format(lmd2)) + " entropy : " + str(h / w))
        out_file.write("\n")