import sys
from collections import defaultdict
import math

lambda1 = 0.95
lambda_unk = 1 - lambda1 #未知語の確率
V = 1000000 #testファイルの未知語を含む語彙数
W = 0 #testファイルの単語数
H = 0 #エントロピー
unk_word = 0 #未知語の数

#各単語の確率を保存する辞書
p_dict = defaultdict(lambda: 0)

#モデル読み込み
model_file = open("train-output.txt", "r").readlines()
for line in model_file:
    words_list = line.strip().split()
    word = words_list[0]
    probability = words_list[1]
    p_dict[word] = probability

#評価
test_file = open(sys.argv[1], "r").readlines()#階層遠いからコマンドラインで指定
for line in test_file:
        test_list = line.strip().split()
        W += 1 #文末記号1カウント

        for word in test_list: #1単語ずつ確率を計算
            W += 1 #単語1カウント
            P = float(lambda_unk) / V

            #未知語かどうか
            if word in p_dict:
                P += lambda1 * float(p_dict[word])
            else:
                unk_word += 1
            H -= math.log2(P)

#結果表示
print("entropy = ",float(H)/W) #エントロピー
print("coverage = ",float(W - unk_word)/W) #カバレージ
