import sys
import math
from collections import defaultdict

class Wordseg:
    def __init__(self):
        self.train_dic = defaultdict(lambda: 0)
        self.lmd = 0.95
        self.V = 1000000
        self.train_result = open(sys.argv[1], "r").readlines()
        self.model_file = open(sys.argv[2], "r").readlines()

    def prepare(self):
        for line in self.train_result:
            line = line.strip()
            words = line.split("\t")
            self.train_dic[words[0]] = float(words[1])

    def positive_step(self, model, train):
        best_score = [0] * (len(model) + 1)
        best_edge = [None] * (len(model) + 1)
        for word_end in range(1, len(model) + 1):
            best_score[word_end] = 10000000000
            for word_begin in range(word_end):
                word = model[word_begin:word_end]
                if word in train.keys() or len(word) == 1:
                    prob = (1 - self.lmd) / self.V
                    if train[word] != 0:
                        prob += self.lmd * float(train[word])
                    my_score = best_score[word_begin] + (-1) * math.log(prob, 2)
                    if my_score < best_score[word_end]:
                        best_score[word_end] = my_score
                        best_edge[word_end] = (word_begin, word_end)

    def negative_step(self, best_edge, model):
        best_path = []
        next_path = best_edge[len(best_edge) - 1]
        words = []
        while next_path != None:
            word = model[next_path[0]:next_path[1]]
            words.append(word)
            next_path = best_edge[next_path[0]]
        words.reverse()
        words = " ".join(words)

    def Viterbi_algorithm(self):
        dic = self.prepare()
        for line in self.model_file:
            line = line.strip()
            best = self.positive_step(line, self.train_dic)
            words = self.negative_step(best_path, line)
            print(words)

if __name__ == "__main__":
    div = Wordseg()
    div.prepare()
    div.Viterbi_algorithm()