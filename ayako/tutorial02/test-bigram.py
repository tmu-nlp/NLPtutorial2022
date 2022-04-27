import sys
import math
import numpy as np
from collections import defaultdict

#lambda1 = 0.10
#lambda2 = 0.15
V = 1000000 #testファイルの未知語を含む語彙数
W = 0 #testファイルの単語数
H = 0 #エントロピー
p_dict = defaultdict(lambda: 0)#各単語の確率を保存する辞書

#モデル読み込み
model_file = open("train-output.txt", "r").readlines()
for line in model_file:
    words_list = line.strip().split()
    if len(words_list) == 2: #1単語の確率
        p_dict[words_list[0]] = words_list[1] #key:単語, value:確率
    else: #2単語の確率
        p_dict[words_list[0] + " " + words_list[1]] = words_list[2]

#出力のヘッダ
print("lambda1\tlambda2\tentropy")

#評価
test_file = open(sys.argv[1], "r").readlines()
for lambda1 in np.arange(0.10, 1.00, 0.05): #rangeの引数に小数点指定したくてnumpy
    for lambda2 in np.arange(0.10, 1.00, 0.05):
        for line in test_file:
            test_list = line.strip().split()
            test_list.append("</s>") #文末記号
            test_list.insert(0,"<s>") #文頭記号

            for i in range(1,len(test_list)):
                word = test_list[i]
                pair = test_list[i-1] + " " + test_list[i]
                P1 = lambda1 * float(p_dict[word]) + (1 - lambda1) / V
                P2 = lambda2 * float(p_dict[pair]) + (1 - lambda2) * P1
                H -= math.log2(P2)
                W += 1 #単語1カウント

        #結果表示
        print('{:.2f}'.format(lambda1),"\t",'{:.2f}'.format(lambda2),"\t",float(H)/W) #エントロピー

#できたら平滑化もしたい