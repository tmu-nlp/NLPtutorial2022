'''
文Ｘが与えられ、構文器Yを予測
PCFG(probabilistic context-free grammar): 各ノードの確率を個別に定義
超グラフの最もスコアの小さい木を探索するには、ビタビアルゴリズムを利用：
    ‐前向きステップ:各ノードまでの最短経路を計算(各ノードの最小部分木のスコアを計算)
    ‐後ろ向きステップ:最短経路を復元(スコア最小の木を復元)
CKY algorithm: チョムスキー標準形文法に基いて、パイパーグラフを構築
'''

import math
from collections import defaultdict

class CKY:
    def __init__(self):
        self.nonterm = []        # (左、右1、右2、確率)の非終端記号
        self.preterm = defaultdict(list)

# p65:文法の読み込み
    def load_grammar(self, grammar_file):
        with open(grammar_file, 'r', encoding='utf-8') as f:
            for rule in f:
                lhs, rhs, prob = rule.strip().split('\t')
                rhs_symbols = rhs.split()
                prob = float(prob)
                if len(rhs_symbols) == 1:    # 前終端記号
                    self.preterm[rhs].append((lhs, math.log(prob)))
                else:          # 非終端記号
                    self.nonterm.append((lhs, rhs_symbols[0], rhs_symbols[1], math.log(prob)))
    # return nonterm, preterm


    def main(self, words):
        # best_score[sym_{i,j}] := 最大対数確率
        self.best_score = defaultdict(lambda : -float('inf'))
        # best_edge[sym_{i,j}] := (lsym_{i,k}, rsym_{k,j})
        self.best_edge = {}
        # 前終端記号を追加
        for i, word in enumerate(words):
            if self.preterm[word]:
                for lhs, log_prob in self.preterm[word]:
                    self.best_score[f'{i}|{i+1}|{lhs}'] = log_prob

        # p67:非終端記号の組み合わせ
        for j in range(2, len(words)+1):
            for i in range(j-2, -1, -1):   # iはスパンの左側(右から左へ処理)
                for k in range(i+1, j):     # kはrsymの開始点
                    # 各文法ルールを展開:log(P(sym -> lsym rsym)) = logprob
                    for sym, lsym, rsym, logprob in self.nonterm:
                        key = f'{i}|{j}|{sym}'
                        l_key = f'{i}|{k}|{lsym}'
                        r_key = f'{k}|{j}|{rsym}'
                        if self.best_score[l_key] > -float('inf') and self.best_score[r_key] > -float('inf'):
                            my_lp = self.best_score[l_key] + self.best_score[r_key] + logprob
                            if my_lp > self.best_score[key]:
                                self.best_score[key] = my_lp
                                self.best_edge[key] = (l_key, r_key)
        return self.create_tree(f'0|{len(words)}|S', words)


    def create_tree(self, key, words):
        sym = key.split('|')[2]
        if key in self.best_edge.keys():
            lkey, rkey = self.best_edge[key]
            lstruct = self.create_tree(lkey,  words)
            rstruct = self.create_tree(rkey, words)
            return f'({sym} {lstruct} {rstruct})\t'
        else:
            i = int(key.split('|')[0])
            return f'({sym} {words[i]})\t'

if __name__ == '__main__':
    cky = CKY()
    grammar_file = '../data/wiki-en-test.grammar'
    cky.load_grammar(grammar_file)
    input_file = '../data/wiki-en-short.tok'
    results_file = 'results_c.txt'
    with open(input_file, 'r', encoding='utf-8') as f, \
        open(results_file, 'w', encoding='utf-8') as f_ans:
        for line in f.readlines():
            # p66:前終端記号を追加
            words = line.strip().split()
            #results = cky.main(words),
            print(cky.main(words), file=f_ans)




