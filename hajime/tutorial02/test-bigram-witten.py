import sys  #i/o
import math
from collections import defaultdict #default-set

model_file = open("model-file2.txt","r").readlines()
# data_file = open("data/wiki-en-train.word","r").readlines()
trg_file = open(sys.argv[1],"r").readlines()
prob = defaultdict(lambda: 0)
counts = defaultdict(lambda: 0)
diff = defaultdict(lambda: 0)
lambda_1 = 0.95
lambda_2 = 0.95
V = 1000000
W = 0
H = 0
lambda_list = list()

#load model
for line in model_file:
    line_list = line.strip()
    pair = line_list.split(" ")
    key = " ".join(pair[0:len(pair)-1])
    prob[key] = pair[-1]


# Witten-Bell-smoothing
# lambda_2をsmoothing
# c(word) : wordの出現回数
for line in trg_file:
    line_list = line.strip()
    words = line_list.split(" ")
    words.insert(0,"<s>")
    words.append("</s>")
    for word in words:
        counts[word] += 1
# print(counts)

# u(word) : wordの次の単語の異なり数
# trg_file中の単語についてbigram_setを作成
bigram_set = set()
for line in trg_file:
    words = line.strip().split(" ")
    words.append("</s>")
    words.insert(0,"<s>")
    for i in range(len(words)):
        bigram = " ".join(words[i-1:i+1])
        bigram_set.add(bigram)

# 得られたbigram_setから異なり語数を獲得
for words in bigram_set:
    word = words.split(" ")
    diff[word[0]] += 1

for line in trg_file:
    words = line.strip().split(" ")
    words.append("</s>")
    words.insert(0,"<s>")
    for i in range(1,len(words)):
        unigram = words[i]
        bigram = " ".join(words[i-1:i+1])
        lambda_2 = 1 - float(diff[words[i-1]])/(diff[words[i-1]] + counts[words[i-1]]) 
        # print(f"diff[words[i-1]] : {diff[words[i-1]]} : {words[i-1]}")
        # print(f"counts[words[i-1]] : {counts[words[i-1]]} : {words[i-1]}")
        P1 = lambda_1 * float(prob[unigram]) + (1-lambda_1) / V
        P2 = lambda_2 * float(prob[bigram]) + (1-lambda_2) * P1
        H += - math.log2(P2)
        W += 1
print(f"entropy : {+float(+H)/W}")