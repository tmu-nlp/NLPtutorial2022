import sys,math
from collections import defaultdict

class Word_Segment:
    def __init__(self):
        self.V = 1000000
        self.lambda1 = 0.950 
        self.p_dict = defaultdict(lambda : 0)

    def train_unigram(self, my_file):
        for line in my_file:
            my_list = line.strip().split()#空白区切りで単語を配列に格納
            my_list.append("</s>")

            for value in my_list:
                self.p_dict[value] += 1

        count = sum(self.p_dict.values())
        for key, value in sorted(self.p_dict.items()):#key:単語，value:確率の辞書
            self.p_dict[key] = '{:.6f}'.format(float(value)/count)

    def forward_step(self,line):
        line = line.strip()#unicode変換いる？文字列長の取得できるやん
        best_edge = [None] * (len(line)+1)
        best_score = [0] * (len(line)+1)

        for word_end in range(1,len(line)+1):
            best_score[word_end] = 10**10#とりあえず大きい値

            for word_begin in range(0,word_end):
                word = line[word_begin:word_end]#部分文字列を取得
                if word in self.p_dict.keys() or len(word) == 1:#既知語か語の長さ1の時
                    P = (1 - self.lambda1) / self.V 
                    if self.p_dict[word]:
                        P += self.lambda1 * float(self.p_dict[word])
                    my_score = best_score[word_begin] - math.log2(P) #確率が大きいほどmy_scoreが小さくなる
                    if my_score < best_score[word_end]:#my_scoreがそこまでの最短経路の時
                        best_score[word_end] = my_score
                        best_edge[word_end] = [word_begin, word_end]
        return best_edge

    def backword_step(self, best_edge, line):
        words = []
        next_edge = best_edge[len(best_edge) - 1]#後ろから
        while next_edge != None:#一番最初のNoneに到達するまで
            #このエッジの部分文字列を追加
            word = line[next_edge[0]:next_edge[1]]
            words.append(word)
            next_edge = best_edge[next_edge[0]]
        words.reverse()
        print(" ".join(words))#join words into a string and print

    def word_segmentation(self, input_file):
        for line in input_file:
            best_edge = self.forward_step(line)#前向きステップ
            output = self.backword_step(best_edge, line)#後ろ向きステップ

if  __name__ == "__main__":
    path = "../../../nlptutorial/"
    my_file = open(path + "data/wiki-ja-train.word", "r").readlines()
    input_file = open(path + "data/wiki-ja-test.txt", "r").readlines()
    x = Word_Segment()
    x.train_unigram(my_file)#unigram辞書つくる
    x.word_segmentation(input_file)#単語分割

"""
Sent Accuracy: 23.81% (20/84)
Word Prec: 68.93% (1861/2700)
Word Rec: 80.77% (1861/2304)
F-meas: 74.38%
Bound Accuracy: 83.25% (2683/3223)
"""