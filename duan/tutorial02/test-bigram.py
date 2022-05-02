import math
from collections import defaultdict

#必要なファイルの読み込み
model_file = open('./nlptutorial/data/train-bigram.txt').readlines()
test_file = open('./nlptutorial/test/02-train-input.txt').readlines()
#初期化
probs = defaultdict(lambda:0)
lambda_1 = 0.95; lambda_2 = 0.05
V = 1000000; W = 0; H = 0

#学習済みのモデルをロードする
for line1 in model_file:
    words1, prob = line1.split('  ')
    probs[words1] = float(prob)

for line2 in test_file:
    words2 = line2.split() #単語を分ける
    words2.append('</s>') #文末記号
    words2.insert(0,'<s>') #文頭記号
    for i in range(1,len(words2)):
        P1 = lambda_1 * probs[words2[i]] + (1-lambda_1)/V  #1-gramの平滑化された確率
        P2 = lambda_2 * probs[words2[i-1]+' '+words2[i]] + (1-lambda_2) * P1  #2-gramの平滑化された確率
        H += -math.log2(P2)
        W += 1 
print('entropy = ' + str(round(H/W,6)))  #エントロピーを計算する
