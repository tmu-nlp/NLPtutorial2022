from math import log, inf
from collections import defaultdict


class Parse:
    def __init__(self, grammar_file, input_file, ans_file):
        self.nonterm = []
        self.preterm = defaultdict(list)
        self.best_edge  = {}
        self.grammar_file = grammar_file
        self.input_file = input_file
        self.ans_file = ans_file

    def loade_grammar(self):
        with open(self.grammar_file, "r") as f:
            for line in f:
                lhs, rhs, prob = line.split("\t")
                prob = float(prob)
                rhs = rhs.split()
                if len(rhs) == 1:
                    self.preterm[rhs[0]].append([lhs, log(prob)])
                else:
                    self.nonterm.append([lhs, rhs[0], rhs[1], log(prob)])

    def CKY(self):
        self.loade_grammar()
        with open(input_file, "r") as data:
            for line in data:
                self.best_score = defaultdict(lambda: -inf)
                self.best_edge = {}

                words = line.strip().split()
                for i in range(len(words)):
                    if words[i] in self.preterm:
                        for lhs, log_prob in self.preterm[words[i]]:
                            self.best_score[f"{lhs}|{i}|{i+1}"] = log_prob
                self.combination_nonterm(words)

        
    def combination_nonterm(self, words):    
        for j in range(2, len(words)+1):
            for i in range(j - 1)[::-1]:
                for k in range(i+1, j):
                    for sym, lsym, rsym, logprob in self.nonterm:
                        if f"{lsym}|{i}|{k}" in self.best_score and f"{rsym}|{k}|{j}" in self.best_score:
                            my_lp = self.best_score[f"{lsym}|{i}|{k}"] + self.best_score[f"{rsym}|{k}|{j}"] + logprob
                            if my_lp > self.best_score[f"{sym}|{i}|{j}"]:
                                self.best_score[f"{sym}|{i}|{j}"] = my_lp
                                self.best_edge[f"{sym}|{i}|{j}"] = (f"{lsym}|{i}|{k}", f"{rsym}|{k}|{j}")
        with open(self.ans_file, "a") as f:
            f.write(self.PRINT(f"S|0|{len(words)}", words))
            f.write("\n")

    def PRINT(self, sym_ij, words):
        sym, i, _ = sym_ij.split("|")
        if sym_ij in self.best_edge:
            return f"({sym} {self.PRINT(self.best_edge[sym_ij][0], words)} {self.PRINT(self.best_edge[sym_ij][1], words)})"
        else:
            i = int(i)
            return f"({sym} {words[i]})"

if __name__ == "__main__":
    gramar_file = "wiki-en-test.grammar"
    input_file = "wiki-en-short.tok"
    ans_file = "wiki-ans"
    parse = Parse(gramar_file, input_file, ans_file)
    with open(ans_file, "w") as f:
        pass
    parse.CKY()