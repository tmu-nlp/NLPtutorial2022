'''
train_file: ../data/wiki-en-train.word
test_file: ../data/wiki-en-test.wordに対して、エントロピーを計算
'''
from collections import defaultdict
import math
import numpy as np


def train_bigram(train_file):
    cnts = defaultdict(int)
    con_cnts = defaultdict(int)
    probs = defaultdict(float)

    with open(train_file, 'r', encoding='utf-8') as f_train:
        for line in f_train:
            words = line.strip().split()
            words.insert(0, '<s>')
            words.append('</s>')
            for i in range(1,len(words)):    # <s>は使わない
                cnts[f'{words[i-1]} {words[i]}'] += 1
                con_cnts[words[i-1]] += 1
                cnts[words[i]] += 1
                con_cnts[''] += 1
    for ngram,cnt in cnts.items():
        words = ngram.split()
        if len(words) == 1:
            probs[ngram] = cnts[ngram]/ con_cnts['']            # unigram
        else:
            probs[ngram] = cnts[ngram] / con_cnts[words[0]]     # bigram
    return probs


# 線形補完を用いて再計算、そしてエントロピーを計算
def test_bigram(test_file, lamb_1, lamb_2, probs):
    V, W, H = 1000000, 0, 0
    with open(test_file, 'r', encoding='utf-8') as f_test:
        for line in f_test:
            words = line.strip().split()
            words.insert(0, '<s>')
            words.append('</s>')
            for i in range(1, len(words)):     # p8
                P1 = lamb_1 * probs[words[i]] + (1-lamb_1)/V
                P2 = lamb_2 * probs[f'{words[i-1]} {words[i]}'] + (1-lamb_2)*P1
                H += -math.log(P2, 2)
                W += 1
        entropy = H / W

    return entropy


def grid_search(test_file, probs):
    min_entropy = float('INF')
    lambda1, lambda2 = 0, 0
    for lamb_1 in np.arange(0.05, 1, 0.05):
        for lamb_2 in np.arange(0.05, 1, 0.05):
            entropy = test_bigram(test_file, lamb_1, lamb_2, probs)
            if entropy < min_entropy:
                min_entropy = entropy
                lambda1 = lamb_1
                lambda2 = lamb_2

    return min_entropy, lambda1, lambda2


if __name__ == '__main__' :
    train_file = '../data/wiki-en-train.word'
    test_file = '../data/wiki-en-test.word'
    probs = train_bigram(train_file)
    results = grid_search(test_file, probs)
    print(f'Entropy:{results[0]}\tlambda1:{results[1]}\tlambda2:{results[2]}')
    # Entropy:9.663869726053504	lambda1:0.8500000000000001	lambda2:0.35000000000000003






