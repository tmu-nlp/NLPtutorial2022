import math
from collections import defaultdict

model_file = open('./nlptutorial/data/train-unigram.txt')
test_file = open('./nlptutorial/data/wiki-en-test.word')
probabilities = defaultdict(lambda:0)
lambda_1 = 0.95; lambda_unk = 1 - lambda_1
V = 1000000; W = 0; H = 0
unknown_word = 0

for line in model_file:
    words = line.split()
    w = words[0]
    P = words[1]
    probabilities[w] = float(P)

for line in test_file:
    words = line.split()
    words.append('</s>')
    for w in words:
        W += 1
        P = lambda_unk / V
        if w in probabilities:
            P += lambda_1 * probabilities[w]
        else:
            unknown_word += 1
        H += -math.log2(P) 
print('entropy = ' + str(round(H / W, 6)))
print('coverage = ' + str(round((W - unknown_word) / W, 6)))
