#python tutorial04.py ../../../nlptutorial/test/05-train-input.txt train_output.txt ../../../nlptutorial/test/05-test-input.txt output.txt
#python tutorial04.py ../../../nlptutorial/data/wiki-en-train.norm_pos wiki-en-model.txt ../../../nlptutorial/data/wiki-en-test.norm my_answer.pos
#perl ../../../nlptutorial/script/gradepos.pl ../../../nlptutorial/data/wiki-en-test.pos my_answer.pos > accuracy_result.txt  
from collections import defaultdict
import sys
import math

class EstimatePOS():
    def __init__(self):
        self.context = defaultdict(lambda: 0) #訓練データでの各品詞のカウント スライドP7で言うと c(NN)
        self.emit = defaultdict(lambda: 0) #訓練データでの品詞と単語の頻度 P7だと c(NN -> language)
        self.transition_count = defaultdict(lambda: 0) #訓練データでの品詞の遷移頻度 P7だと c(NN LRB)
        
        self.transition_p = defaultdict(lambda: 0) #モデルファイルから読み込んだ品詞の遷移確率
        self.emission = defaultdict(lambda: 0) #モデルファイルから読み込んだ品詞に対する単語の確率
        self.possible_tags = defaultdict(lambda: 0) #モデルファイルから読み込んだモデル内で出てくる品詞（tag）#リストでもいいんじゃない？
        self.lambda_1 = 0.95
        self.v = 1000000


    def calcu_hmm_model(self, train_filename, model_filename):
        with open(train_filename, "r") as f_train:
            data = f_train.readlines()
        for line in data:
            line = line.strip().split()
            previous = "<s>"
            self.context[previous] += 1
            for wordtag in line:
                wordtag = wordtag.split("_")
                word = wordtag[0]
                tag = wordtag[1]
                self.transition_count[previous + " " + tag] += 1
                self.context[tag] += 1
                self.emit[tag + " " + word] += 1
                previous = tag
            self.transition_count[previous + " " + "</s>"] += 1
        
        #上で頻度をカウント　それらを使ってこっから下で確率を計算
        with open(model_filename, "w") as f_model:
            for my_key, my_value in sorted(self.transition_count.items()):
                tag_tag = my_key.split(" ")
                f_model.write(f'T {my_key} {my_value / self.context[tag_tag[0]]}\n')

            for my_key, my_value in sorted(self.emit.items()):
                tag_word = my_key.split(" ")
                f_model.write(f'E {my_key} {my_value / self.context[tag_word[0]]}\n')

        
    def load_model(self, model_filename):
        with open(model_filename, "r") as f_model:
            data = f_model.readlines()
        for line in data:
            line = line.strip().split()
            self.possible_tags[line[1]] = 1
            if line[0] == "T":
                self.transition_p[f'{line[1]} {line[2]}'] = float(line[3])
            else:
                self.emission[f'{line[1]} {line[2]}'] = float(line[3])


    def estimate_pos(self, input_filename, out_filename): #前向きステップと後ろ向きステップの合体
        with open(input_filename, "r") as f_input:
            data = f_input.readlines()
        with open(out_filename, "w") as f_output:
            for line in data:
                line = line.strip()
                best_edge = self.front_step(line)
                ans_str = self.back_step(line, best_edge)
                f_output.write(f'{ans_str}\n')


    def front_step(self, line): #前向きステップ
        line = line.split()
        best_score = defaultdict(lambda: 0)
        best_edge = dict()
        best_score["0 <s>"] = 0
        best_edge["0 <s>"] = None
        for i in range(len(line)):
            for prev_tag in self.possible_tags.keys():
                for next_tag in self.possible_tags.keys():
                    if (f'{i} {prev_tag}' in best_score) and (f'{prev_tag} {next_tag}' in self.transition_p):
                        #平滑化
                        p = self.lambda_1 * self.emission[f'{next_tag} {line[i]}'] + (1 - self.lambda_1) / self.v
                        score = best_score[f'{i} {prev_tag}'] + -math.log2(self.transition_p[f'{prev_tag} {next_tag}']) + -math.log2(p)

                        if (f'{i+1} {next_tag}' not in best_score) or (best_score[f'{i+1} {next_tag}'] > score):
                            best_score[f'{i+1} {next_tag}'] = score
                            best_edge[f'{i+1} {next_tag}'] = f'{i} {prev_tag}'
        
        #文末</s>にも同じ処理を
        for prev_tag in self.possible_tags:
            if (f'{len(line)} {prev_tag}' in best_score) and (f'{prev_tag} </s>' in self.transition_p):
                score = best_score[f'{len(line)} {prev_tag}'] + -math.log2(self.transition_p[f'{prev_tag} </s>']) #</s>は品詞(tag)として保存されてるからlog2(p)はいらない(と思う)

                if (f'{len(line) + 1} </s>' not in best_score) or (best_score[f'{len(line) + 1} </s>'] > score):
                    best_score[f'{len(line) + 1} </s>'] = score
                    best_edge[f'{len(line) + 1} </s>'] = f'{len(line)} {prev_tag}'
        return best_edge


    def back_step(self, line, best_edge): #後ろ向きステップ
        tags = []
        line = line.split()
        next_edge = best_edge[f'{len(line) + 1} </s>']
        while next_edge != "0 <s>":
            num_and_tag = next_edge.split(" ")
            tags.append(num_and_tag[1])
            next_edge = best_edge[next_edge]
        tags.reverse()
        return " ".join(tags)


if __name__ == "__main__":
    k = EstimatePOS()
    k.calcu_hmm_model(sys.argv[1], sys.argv[2])
    k.load_model(sys.argv[2])
    k.estimate_pos(sys.argv[3], sys.argv[4])