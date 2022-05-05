import sys,math
from collections import defaultdict

#load 1gram_dict
def train_unigram(my_file):
    my_dict = defaultdict(lambda: 0)#頻度カウント用辞書
    p_dict = defaultdict(lambda : 0)#unigram確率用辞書
    for line in my_file:
        my_list = line.strip().split()#空白区切りで単語を配列に格納
        my_list.append("</s>")
        for value in my_list:
            my_dict[value] += 1
    count = sum(my_dict.values())
    for key, value in sorted(my_dict.items()):#key:単語，value:確率の辞書
        p_dict[key] = '{:.6f}'.format(float(value)/count)

    return p_dict

def word_segmentation(p_dict, input_file, V, lambda1):
    for line in input_file:
        #前向きステップ
        line = line.strip()#unicode変換いる？文字列長の取得できるやん
        best_edge = [None] * (len(line)+1)
        best_score = [0] * (len(line)+1)

        for word_end in range(1,len(line)+1):
            best_score[word_end] = 10**10#とりあえず大きい値

            for word_begin in range(0,word_end):
                word = line[word_begin:word_end]#部分文字列を取得
                if word in p_dict.keys() or len(word) == 1:#既知語か語の長さ1の時
                    P = (1 - lambda1) / V 
                    if p_dict[word]:
                        P += lambda1 * float(p_dict[word])
                    my_score = best_score[word_begin] - math.log2(P) #確率が大きいほどmy_scoreが小さくなる
                    if my_score < best_score[word_end]:#my_scoreがそこまでの最短経路の時
                        best_score[word_end] = my_score
                        best_edge[word_end] = [word_begin, word_end]
        #後ろ向きステップ
        words = []
        next_edge = best_edge[len(best_edge) - 1]#後ろから
        while next_edge != None:#一番最初のNoneに到達するまで
            #このエッジの部分文字列を追加
            word = line[next_edge[0]:next_edge[1]]
            words.append(word)
            next_edge = best_edge[next_edge[0]]
        words.reverse()
        print(" ".join(words))#join words into a string and print

    return 0

V = 1000000
lambda1 = 0.950 
path = "../../../nlptutorial/"
my_file = open(path + "data/wiki-ja-train.word", "r").readlines()
input_file = open(path + "data/wiki-ja-test.txt", "r").readlines()
p_dict = train_unigram(my_file)


x = word_segmentation(p_dict, input_file, V, lambda1)

"""
Sent Accuracy: 2.38% (2/84)
Word Prec: 2.38% (2/84)
Word Rec: 0.09% (2/2307)
F-meas: 0.17%
Bound Accuracy: 31.09% (1003/3226)

ちょっと低すぎる気がする
"""