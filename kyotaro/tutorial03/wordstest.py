import sys
import math
from collections import defaultdict

train_result = open(sys.argv[1], "r").readlines()
model_file = open(sys.argv[2], "r").readlines()
lmd = 0.95
V = 1000000

### prepare ###
train = defaultdict(lambda: 0)
for line in train_result:
    line = line.strip()
    words = line.split()
    train[words[0]] = float(words[1])

### Viterbi algorithm ###
for line in model_file:
    line = line.strip()
    ### positive step ###
    best_score = [0] * (len(line) + 1)
    best_edge = [None] * (len(line) + 1)
    for word_end in range(1, len(line) + 1):
        best_score[word_end] = 10000000000
        for word_begin in range(word_end):
            word = line[word_begin:word_end]
            if word in train.keys() or len(word) == 1:
                prob = (1 - lmd) / V
                if train[word] != 0:
                    prob += lmd * float(train[word])
                my_score = best_score[word_begin] + (-1) * math.log(prob, 2)
                if my_score < best_score[word_end]:
                    best_score[word_end] = my_score
                    best_edge[word_end] = (word_begin, word_end)
    ### negative step ###
    best_path = []
    next_path = best_edge[len(best_edge) - 1]
    words = []
    while next_path != None:
        word = line[next_path[0]:next_path[1]]
        words.append(word)
        next_path = best_edge[next_path[0]]
    words.reverse()
    words = " ".join(words)
    print(words)