import math
from collections import defaultdict

class test_HMM:
    def __init__(self):
        self.V = 1000000
        self.lambda1 = 0.950
        self.transition = defaultdict(lambda : 0)
        self.emission = defaultdict(lambda : 0)
        self.possible_tags = defaultdict(lambda : 0)

    #モデル読み込み
    def load_model(self, model_file):
        for line in model_file:
            #T X X 0.333333
            model_list = line.split()#type, context, word, probの順で配列に格納
            self.possible_tags[model_list[1]] = 1#可能なタグとして保存
            if model_list[0] == "T":#type = 遷移確率
                #前一個の単語を考慮した確率
                self.transition[model_list[1] + " " + model_list[2]] = float(model_list[3])
            else:#type = 生成確率
                self.emission[model_list[1] + " " + model_list[2]] = float(model_list[3])

    #前向きステップ
    def forward_step(self, line):
        words = line.strip().split()
        l = len(words)
        best_score = defaultdict(lambda : 0)
        best_edge = defaultdict(lambda : 0)
        best_score["0 <s>"] = 0
        best_edge["0 <s>"] = None
        for i in range(l):
            for prev in self.possible_tags.keys():
                for next_tag in self.possible_tags.keys():
                    if str(i)+" "+prev in best_score and prev+" "+next_tag in self.transition:
                        P_T = self.transition[prev+" "+next_tag]
                        P_E = self.lambda1 * float(self.emission[next_tag+" "+words[i]]) + (1 - self.lambda1) / self.V
                        score = best_score[str(i)+" "+prev] - math.log2(P_T) - math.log2(P_E)
                        if str(i+1)+" "+next_tag not in best_score or best_score[str(i+1)+" "+next_tag] > score:#新しいやつの方が最小の時
                            best_score[str(i+1)+" "+next_tag] = score
                            best_edge[str(i+1)+" "+next_tag] = str(i)+ " "+prev
                        
        #文末に同じ処理
        for tag in self.possible_tags.keys():
            if str(l)+" "+tag in best_score and tag+" </s>" in self.transition:
                P_T = self.transition[tag+" </s>"]
                score = best_score[str(l)+" "+tag]  - math.log2(P_T)
                if str(l+1)+" </s>" not in best_score or best_score[str(l+1)+" </s>"] > score:#新しいやつの方が最小の時
                    best_score[str(l+1)+" </s>"] = score
                    best_edge[str(l+1)+" </s>"] = str(l)+ " "+tag
        return best_edge,l

    #後ろ向きステップ
    def backward_step(self, best_edge, line, l):
        tags = []
        next_edge = best_edge[str(l+1)+" </s>"]#一番後ろの
        while next_edge != "0 <s>":#文頭にくるまで
            #このエッジの品詞を出力に追加
            posi_and_tag = next_edge.split()
            tags.append(posi_and_tag[1])#tag append to tags
            next_edge = best_edge[next_edge]
        tags.reverse()
        print(" ".join(tags))#join tags into a string and print
        

    def viterbi(self, test_file):
        for line in test_file:
            best_edge,l = self.forward_step(line)#前向きステップ
            output = self.backward_step(best_edge, line, l)#後ろ向きステップ

if  __name__ == "__main__":
    path = "../../../nlptutorial/"
    """
    テスト用
    model_file = open("train-output.txt", "r").readlines()
    test_file = open(path + "test/05-test-input.txt", "r").readlines()
    """
    model_file = open("train-model.txt", "r").readlines()
    test_file = open(path + "data/wiki-en-test.norm", "r").readlines()
    x = test_HMM()
    x.load_model(model_file)
    x.viterbi(test_file)
    
"""
Accuracy: 90.82% (4144/4563)

Most common mistakes:
NNS --> NN      45
NN --> JJ       27
JJ --> DT       22
NNP --> NN      22
VBN --> NN      12
JJ --> NN       12
NN --> IN       11
NN --> DT       10
NNP --> JJ      8
VBN --> JJ      7
"""