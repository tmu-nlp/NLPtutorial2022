import sys
import math
from collections import defaultdict

def prepare(train):
    train_dic = defaultdict(lambda: 0)
    for line in train_result:
        line = line.strip()
        words = line.split()
        train_dic[words[0]] = float(words[1])
    return train_dic

def positive_step(model, train, lmd, V):
    best_score = [0] * (len(model) + 1)
    best_edge = [None] * (len(model) + 1)
    for word_end in range(1, len(model) + 1):
        best_score[word_end] = 10000000000
        for word_begin in range(word_end):
            word = model[word_begin:word_end]
            if word in train.keys() or len(word) == 1:
                prob = (1 - lmd) / V
                if train[word] != 0:
                    prob += lmd * float(train[word])
                my_score = best_score[word_begin] + (-1) * math.log(prob, 2)
                if my_score < best_score[word_end]:
                    best_score[word_end] = my_score
                    best_edge[word_end] = (word_begin, word_end)
    return best_edge

def negative_step(best_edge, model):
    best_path = []
    next_path = best_edge[len(best_edge) - 1]
    words = []
    while next_path != None:
        word = model[next_path[0]:next_path[1]]
        words.append(word)
        next_path = best_edge[next_path[0]]
    words.reverse()
    words = " ".join(words)
    return words

def Viterbi_algorithm(model_file, train_result, lmd, V):
    dic = prepare(train_result)
    for line in model_file:
        line = line.strip()
        best = positive_step(line, dic, lmd, V)
        words = negative_step(best, line)
        print(words)

train_result = open(sys.argv[1], "r").readlines()
model_file = open(sys.argv[2], "r").readlines()
lmd = 0.95
V = 1000000
Viterbi_algorithm(model_file, train_result, lmd, V)