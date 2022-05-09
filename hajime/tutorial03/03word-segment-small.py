# NLPtutorial 03にて作成

import sys
import math
from collections import defaultdict  # default-set

# input prob
lambda_1 = 0.95
V = 1000000

# prob_file = open("./test/04-model.txt", "r").readlines()
prob_file = open("./output/model-file-ja.txt", "r")
prob_list = defaultdict(lambda: 0)
for line in prob_file:
    line_list = line.strip()
    pair = line_list.split()
    prob_list[pair[0]] = pair[1]
    # print(pair[0], pair[1])

# for key in prob_list.keys():
#     print(key)

# 読み込みはOK

# input_file = open("test/04-input.txt", "r").readlines()
input_file = open("data/wiki-ja-test.txt", "r")

# output = open("04-output.txt", "w")
output = open("my_answer.word", "w")

for line in input_file:
    best_edge = [0] * (len(line)+1)
    best_score = [0] * (len(line)+1)
    best_edge[0] = None
    best_score[0] = 0
    for word_end in range(1, len(line)+1):
        best_score[word_end] = 10**10
        for word_begin in range(0, word_end):
            word = line[word_begin:word_end]
            # print(word)
            if word in prob_list.keys() or len(word) == 1:
                # print(word)
                prob = (1 - lambda_1) / V
                if prob_list[word] != 0:
                    prob += lambda_1 * float(prob_list[word])
                my_score = best_score[word_begin] + (-math.log(prob, 2))
                if my_score < best_score[word_end]:
                    best_score[word_end] = my_score
                    best_edge[word_end] = (word_begin, word_end)
                # print(word, my_score)
    words = []
    next_edge = best_edge[len(best_edge)-1]
    while next_edge != None:
        word = line[next_edge[0]:next_edge[1]]
        words.append(word)
        next_edge = best_edge[next_edge[0]]
    words.reverse()
    output.write(' '.join(words))
