#python tutorial03.py ../../../nlptutorial/data/wiki-ja-train.word ../../../nlptutorial/data/wiki-ja-test.txt my_answer2.word
#perl gradews.pl ../../../nlptutorial/data/wiki-ja-test.word my_answer2.word > split_accuracy.txt
#python tutorial03.py ../../../nlptutorial/data/big-ws-model.txt ../../../nlptutorial/data/wiki-ja-test.txt my_answer_big.word
#perl gradews.pl ../../../nlptutorial/data/wiki-ja-test.word my_answer_big.word > split_accuracy_big.txt
from collections import defaultdict
import math
import sys

class SplitWord:
    def __init__(self):
        self.dict_uni_prob = defaultdict(lambda: 0)
        self.lambda_1 = 0.95
        self.v = 1000000

    
    def calcu_uni_prob(self, train_filename): #1-gramモデルの作成
        with open(train_filename, "r") as f_train:
            data = f_train.readlines()
        for line in data:
            line = line.strip().split()
            line.append('</s>')
            for word in line:
                self.dict_uni_prob[word] += 1
        cnt = sum(self.dict_uni_prob.values())
        for my_key, my_value in sorted(self.dict_uni_prob.items()):
            #self.dict_uni_prob[my_key] = my_value / sum(self.dict_uni_prob.values())
            self.dict_uni_prob[my_key] = my_value / cnt


    def load_uni_prob(self, train_filename): #1-gramモデルの読み込み
        with open(train_filename, "r") as f_train:
            data = f_train.readlines()
        for line in data:
            line = line.strip().split("\t")
            self.dict_uni_prob[line[0]] = float(line[1])


    def split_word(self, input_filename, out_filename): #前向きステップと後ろ向きステップの合体
        with open(input_filename, "r") as f_input:
            data = f_input.readlines()
        with open(out_filename, "w") as f_output:
            for line in data:
                line = line.strip()
                best_edge = self.front_step(line)
                ans_str = self.back_step(line, best_edge)
                f_output.write(f'{ans_str}\n')


    def front_step(self, line): #前向きステップ
        best_score = [10000000] * (len(line) + 1)
        best_edge = [None] * (len(line) + 1)
        best_score[0] = 0

        for word_end in range(1, len(line) + 1):
            for word_begin in range(word_end):
                word = line[word_begin:word_end]
                if len(word) == 1 or word in self.dict_uni_prob.keys():
                    prob = self.lambda_1 * self.dict_uni_prob[word] + (1 - self.lambda_1) / self.v #defauldictだからコメントアウト部分4行はこの1行でおけ

                    my_score = best_score[word_begin] + -math.log(prob)
                    if my_score < best_score[word_end]:
                        best_score[word_end] = my_score
                        best_edge[word_end] = (word_begin, word_end) #best_edgeの中の各要素はタプル型
        
        return best_edge


    def back_step(self, line, best_edge): #後ろ向きステップ
        words = []
        next_edge = best_edge[len(best_edge) - 1]
        while next_edge != None:
            word = line[next_edge[0]:next_edge[1]]
            words.append(word)
            next_edge = best_edge[next_edge[0]]
        words.reverse()
        return " ".join(words)


if __name__ == "__main__":
    k = SplitWord()
    k.calcu_uni_prob(sys.argv[1])
    #k.load_uni_prob(sys.argv[1]) #big-ws-model.txtは既にモデル学習済み
    k.split_word(sys.argv[2], sys.argv[3])
