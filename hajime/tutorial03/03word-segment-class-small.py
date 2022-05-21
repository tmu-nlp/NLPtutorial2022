# NLPtutorial 03にて作成
# 榎本くんのスクリプトを参考に作成

import math
from collections import defaultdict  # default-set


class WordSegment:
    def __init__(self):
        self.dict_uni_prob = defaultdict(lambda: 0)
        self.lambda_1 = 0.95
        self.V = 1000000

    def calc_uni_prob(self, train_file_name):
        total_cnt = 0
        with open(train_file_name, 'r') as train_file:
            data = train_file.readlines()
        for line in data:
            line = line.strip().split()
            line.append('<\s>')
            for word in line:
                total_cnt += 1
                self.dict_uni_prob[word] += 1
        for key, value in sorted(self.dict_uni_prob.items()):
            self.dict_uni_prob[key] = value / total_cnt
            # print(key, self.dict_uni_prob[key])

    def input_uni_prob(self, prob_file_name):
        with open(prob_file_name, 'r') as prob_file:
            for line in prob_file:
                line_list = line.strip().split()
                self.dict_uni_prob[line_list[0]] = line_list[1]

    def front_step(self, line):
        line = line.strip()
        best_edge = [0] * (len(line)+1)
        best_score = [0] * (len(line)+1)
        best_edge[0] = None
        best_score[0] = 0
        for word_end in range(1, len(line)+1):
            best_score[word_end] = 10**10
            for word_begin in range(0, word_end):
                word = line[word_begin:word_end]
                if word in self.dict_uni_prob.keys() or len(word) == 1:
                    prob = (1 - self.lambda_1) / self.V
                    if self.dict_uni_prob[word] != 0:
                        prob += self.lambda_1 * float(self.dict_uni_prob[word])
                    my_score = best_score[word_begin] + \
                        (-math.log(float(prob), 2))
                    if my_score < best_score[word_end]:
                        best_score[word_end] = my_score
                        best_edge[word_end] = (word_begin, word_end)
        return best_edge

    def back_step(self, line, best_edge):
        words = []
        next_edge = best_edge[len(best_edge)-1]
        while next_edge != None:
            word = line[next_edge[0]:next_edge[1]]
            words.append(word)
            next_edge = best_edge[next_edge[0]]
        words.reverse()
        return ' '.join(words)

    def split_word(self, input_file_name, output_file_name):
        with open(input_file_name, 'r') as f_input:
            data = f_input.readlines()
        with open(output_file_name, 'w') as f_output:
            for line in data:
                best_edge = self.front_step(line)
                trg_str = self.back_step(line, best_edge)
                f_output.write(f"{trg_str}\n")


if __name__ == "__main__":
    model_file = "./data/wiki-ja-train.word"
    input_file = "./data/wiki-ja-test.txt"
    output_file = "./my_answer-class.word"

    word_seg = WordSegment()
    word_seg.calc_uni_prob(model_file)
    word_seg.split_word(input_file, output_file)
