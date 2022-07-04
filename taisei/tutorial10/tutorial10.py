from collections import defaultdict
from tqdm import tqdm
import math
from nltk.tree import Tree


class CKY:
    def __init__(self):
        self.nonterm = list()  # (左,右1,右2,確率)の非終端記号
        self.preterm = defaultdict(list)  # preterm[右] = [(左,確率)...]形式のマップ
        self.best_score = defaultdict(
            lambda: -float("inf"))  # [sym_ij] = 最大対数確率
        self.best_edge = dict()  # [sym_ij] = (lsym_ik, rsym_kj)

    def load_grammar(self, grammar_file):
        """文法の読み込み"""
        with open(grammar_file, "r") as f_grammar:
            data_grammar = f_grammar.readlines()

        for rule in data_grammar:
            lhs, rhs, prob = rule.strip().split("\t")
            rhs_symbols = rhs.split()
            prob = float(prob)
            if len(rhs_symbols) == 1:  # 前終端記号
                self.preterm[rhs].append((lhs, math.log(prob)))
            else:  # 非終端記号
                self.nonterm.append(
                    (lhs, rhs_symbols[0], rhs_symbols[1], math.log(prob)))

    def cky_algo(self, input_file):
        with open(input_file, "r") as f_input:
            data = f_input.readlines()
        for line in data:
            self.best_score = defaultdict(lambda: -float("inf"))
            self.best_edge = dict()
            words = line.strip().split()
            self.add_front_final_sym(words)
            self.combi_final_sym(words)
            tree_line = self.output_tree_line(f"S 0 {len(words)}", words)
            print(tree_line)
            self.print_tree(tree_line)
            print('\n')

    def add_front_final_sym(self, words):
        """前終端記号を追加"""
        for i in range(len(words)):
            if words[i] in self.preterm:
                for lhs, log_prob in self.preterm[words[i]]:
                    self.best_score[f"{lhs} {i} {i+1}"] = log_prob

    def combi_final_sym(self, words):
        """非終端記号の組み合わせ"""
        for j in range(2, len(words)+1):  # j:スパンの右側
            for i in range(j-1)[::-1]:  # i:スパンの左側
                for k in range(i+1, j):  # k:rsymの開始点
                    # 文法ルールを展開　log(P(sym->lsym rsym))=logprob
                    for sym, lsym, rsym, log_prob in self.nonterm:
                        left_p = self.best_score[f"{lsym} {i} {k}"]
                        right_p = self.best_score[f"{rsym} {k} {j}"]
                        if left_p > -float("inf") and right_p > -float("inf"):
                            my_lp = left_p + right_p + log_prob

                            if my_lp > self.best_score[f"{sym} {i} {j}"]:
                                self.best_score[f"{sym} {i} {j}"] = my_lp
                                self.best_edge[f"{sym} {i} {j}"] = (
                                    f"{lsym} {i} {k}", f"{rsym} {k} {j}")

    def output_tree_line(self, sym_ij, words):
        """木を出力"""
        sym, i, j = sym_ij.split()
        i, j = int(i), int(j)
        if sym_ij in self.best_edge:
            return f"({sym} {self.output_tree_line(self.best_edge[sym_ij][0], words)} {self.output_tree_line(self.best_edge[sym_ij][1], words)})"
        else:
            return f"({sym} {words[i]})"

    def print_tree(self, tree_line):
        """木構造をプリント"""
        t = Tree.fromstring(tree_line)
        t.pretty_print()


if __name__ == "__main__":
    cky = CKY()
    path = "../../../nlptutorial/data/"
    cky.load_grammar(f'{path}wiki-en-test.grammar')
    cky.cky_algo(f'{path}wiki-en-short.tok')
