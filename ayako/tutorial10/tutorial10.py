from collections import defaultdict
import math, sys

class CKY:
    def __init__(self):
        self.nonterm = [] #(左,右1,右2,確率)の非終端記号
        self.preterm = defaultdict(lambda:[]) #pre[右]=[(左,確率)...]形式のマップ
        self.best_score = defaultdict(lambda:-math.inf) #引数=sym_i,j 値=最大対数確率
        self.best_edge = {} #引数=sym_i,j 値=(lsym_i,k, rsym_k,j)

    def load_grammar(self, grammar_file):
        """lhs \t rhs \t prob \n形式の文法を読み込む"""
        with open(grammar_file, "r") as f:
            for rule in f:
                lhs, rhs, prob = rule.strip().split("\t") #P(左->右)=確率
                prob = float(prob)
                rhs_symbols = rhs.split(" ") #右
                if len(rhs_symbols) == 1: #前終端記号
                    self.preterm[rhs].append((lhs, math.log(prob)))
                else: #非終端記号
                    self.nonterm.append([lhs, rhs_symbols[0], rhs_symbols[1], math.log(prob)])

    def cky_algo(self, input_file, output_file):
        #前終端記号を追加
        with open(input_file, "r") as input_f, open(output_file, "w") as output_f:
            for line in input_f:
                words = line.strip().split(" ")
                for i in range(len(words)):
                    if self.preterm[words[i]]:
                        for lhs, log_prob in self.preterm[words[i]]:
                            self.best_score[f"{lhs} {i} {i+1}"] = log_prob
            #非終端記号の組み合わせ
                for j in range(2, len(words)+1): #jはスパンの右側
                    for i in range(j-2, -1, -1): #iはスパンの左側(右から左へ処理)
                        for k in range(i+1, j): #kはrsym の開始点
                            for sym, lsym, rsym, log_prob in self.nonterm:
                                #各文法ルールを展開:log(P(sym→lsym rsym))=logprob
                                sym_i_j = f"{sym} {i} {j}"
                                lsym_i_k = f"{lsym} {i} {k}"
                                rsym_k_j = f"{rsym} {k} {j}"
                                #両方の子供の確率が0より大きい
                                if self.best_score[lsym_i_k] > -math.inf and self.best_score[rsym_k_j] > -math.inf:
                                    #このノード・辺の対数確率を計算
                                    my_lp = self.best_score[lsym_i_k] + self.best_score[rsym_k_j] + log_prob
                                    #この辺が確率最大のものなら更新
                                    if my_lp > self.best_score[sym_i_j]:
                                        self.best_score[sym_i_j] = my_lp
                                        self.best_edge[sym_i_j] = (lsym_i_k, rsym_k_j)
                
                print(self.print_tree(f"S 0 {len(words)}", words), file=output_f)

    def print_tree(self, sym_i_j, words):
        """再帰で木構造を出力"""
        sym, i, j = sym_i_j.split(" ")
        if sym_i_j in self.best_edge: #非終端記号
            return "("+sym+" "+self.print_tree(self.best_edge[sym_i_j][0], words)+" "+self.print_tree(self.best_edge[sym_i_j][1], words)+")"
        else:
            return "("+sym+" "+words[int(i)]+")"

if __name__ == "__main__":
    path = "../../../nlptutorial/"
    if sys.argv[1] == "test":
        input_file = path + "test/08-input.txt"
        grammar = path + "test/08-grammar.txt"
        output_file = "output_test.txt"
    else:
        input_file = path + "data/wiki-en-short.tok"
        grammar = path + "data/wiki-en-test.grammar"
        output_file = "output.txt"

    cky = CKY()
    cky.load_grammar(grammar)
    cky.cky_algo(input_file, output_file)