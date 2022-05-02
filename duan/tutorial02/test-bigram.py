import sys
import math
from collections import defaultdict

model_file = open('./nlptutorial/data/train-bigram.txt').readlines()
test_file = open('./nlptutorial/test/02-train-input.txt').readlines()
probs = defaultdict(lambda:0)
lambda_1 = 0.95; lambda_2 = 0.95
V = 1000000; W = 0; H = 0

for line1 in model_file:
    words1, prob = line1.split('  ')
    probs[words1] = float(prob)

for line2 in test_file:
    words2 = line2.split()
    words2.append('</s>')
    words2.insert(0,'<s>')
    for i in range(1,len(words2)):
        P1 = lambda_1 * probs[words2[i]] + (1-lambda_1)/V  # 1-gram の平滑化された確率
        P2 = lambda_2 * probs[words2[i-1]+' '+words2[i]] + (1-lambda_2) * P1  # 2-gram の平滑化された確率
        H += -math.log2(P2)
        W += 1 
print('entropy = ' + str(round(H/W,6)))