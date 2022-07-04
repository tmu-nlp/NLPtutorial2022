from collections import defaultdict
import math
from nltk.tree import Tree
from nltk.tree import TreePrettyPrinter


class CKY():
    def __init__(self, grammar_file):
        self.nonterm = []
        self.preterm = defaultdict(list)
        self.best_score = defaultdict(lambda: -math.inf)
        self.best_edge = {}
        self.read_grammar(grammar_file)

    def read_grammar(self, grammar_file):
        with open(grammar_file, "r") as g_file:
            for rule in g_file:
                lhs, rhs, prob = rule.strip().split("\t")
                rhs_symbols = rhs.split(" ")
                if len(rhs_symbols) == 1:
                    self.preterm[rhs].append((lhs, math.log(float(prob))))
                else:
                    self.nonterm.append(
                        (lhs, rhs_symbols[0], rhs_symbols[1], math.log(float(prob))))

    def cky_algorithm(self, input_file):
        with open(input_file, "r") as i_file:
            for line in i_file:
                self.best_score = defaultdict(lambda: -math.inf)
                self.best_edge = {}
                words = line.strip().split(" ")
                for i in range(len(words)):
                    if self.preterm[words[i]]:
                        for lhs, log_prob in self.preterm[words[i]]:
                            self.best_score[f"{lhs} {i} {i+1}"] = log_prob
                self.combination_nonterminal(words)
                tree_line = self.print_tree(f"S 0 {len(words)}", words)
                t = Tree.fromstring(tree_line)
                print(TreePrettyPrinter(t).text())
                print(tree_line)

    def combination_nonterminal(self, words):
        for j in range(2, len(words)+1):
            for i in range(j-2, -1, -1):
                for k in range(i+1, j):
                    for sym, lsym, rsym, log_prob in self.nonterm:
                        lsym_i_k = f"{lsym} {i} {k}"
                        rsym_k_j = f"{rsym} {k} {j}"
                        left = self.best_score[lsym_i_k]
                        right = self.best_score[rsym_k_j]
                        if left > -math.inf and right > -math.inf:
                            my_lp = left + right + log_prob
                            sym_i_j = f"{sym} {i} {j}"
                            if my_lp > self.best_score[sym_i_j]:
                                self.best_score[sym_i_j] = my_lp
                                self.best_edge[sym_i_j] = (lsym_i_k, rsym_k_j)

    def print_tree(self, sym_i_j, words):
        sym, i, j = sym_i_j.split(" ")
        if sym_i_j in self.best_edge:
            tree1 = "(" + sym + " " + self.print_tree(
                self.best_edge[sym_i_j][0], words) + " " + self.print_tree(self.best_edge[sym_i_j][1], words) + ")"
            return tree1
        else:
            tree2 = "(" + sym + " " + words[int(i)] + ")"
            return tree2


if __name__ == "__main__":

    grammar_file1 = "test/08-grammar.txt"
    input_file1 = "test/08-input.txt"
    cky1 = CKY(grammar_file1)
    cky1.cky_algorithm(input_file1)

    # grammar_file2 = "data/wiki-en-test.grammar"
    # input_file2 = "data/wiki-en-short.tok"
    # cky2 = CKY(grammar_file2)
    # cky2.cky_algorithm(input_file2)
