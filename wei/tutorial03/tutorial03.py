'''
train_file: ../data/wiki-ja-train.word to get unigram probs
test_file: ../data/wiki-ja-test.txt
下のスクリプトを用いて分割精度を評価：
script/gradews.pl data/wiki-ja-test.word my_answer.word
F値を報告
'''

from collections import defaultdict
import math
import subprocess


# get unigram probs
def unigram(train_file):
    cnts = defaultdict(float)
    total_cnts = 0
    with open(train_file, 'r' ,encoding='utf-8') as f_train:
        for line in f_train:
            words = line.strip().split()
            # words = list(''.join(words))
            words.append("</s>")
            for word in words:
                cnts[word] += 1
                total_cnts += 1

    probs = {}
    for w,cnt in cnts.items():
        prob = cnt/total_cnts
        probs[w] = prob

    return probs

def word_seg(test_file, probs, ans_file):
    with open(test_file, 'r', encoding='utf-8') as f_test:
        lines = f_test.readlines()
    with open(ans_file, 'w', encoding='utf-8') as f_ans:
        for line in lines:
            line = line.strip()
            # forward step
            lambda_1, N = 0.95, 1000000
            best_edge = {}
            best_score = {}
            best_edge[0] = None
            best_score[0] = 0
            for word_end in range(1, len(line)+1):
                best_score[word_end] = float('inf')
                for word_begin in range(word_end):
                    word = line[word_begin:word_end]
                    if word in probs.keys() or len(word) == 1:
                        if word in probs.keys():
                            prob = lambda_1 * float(probs[word]) + (1-lambda_1)/N
                        else:
                            prob = (1-lambda_1)/N

                        my_score = best_score[word_begin] - math.log(prob, 2)
                        if my_score < best_score[word_end]:
                            best_score[word_end] = my_score
                            best_edge[word_end] = [word_begin, word_end]

            # backward step
            words = []
            next_edge = best_edge[len(best_edge)-1]
            while next_edge != None:
                word = line[next_edge[0]: next_edge[1]]
                words.append(word)
                next_edge = best_edge[next_edge[0]]

            words.reverse()

            f_ans.write(f"{' '.join(words)} \n")

if __name__ == '__main__':
    # for train-and-seg
    train_file = '../data/wiki-ja-train.word'
    test_file = '../data/wiki-ja-test.txt'
    output_file = 'results.txt'
    probs = unigram(train_file)
    # print(probs['彩'])      -> KeyError
    word_seg(test_file, probs, output_file)


    # for eval
    test_input = '../test/04-input.txt'
    test_model = '../test/04-model.txt'
    my_test_res = 'my_test_results.txt'
    test_answer = '../test/04-answer.txt'

    # load test probs
    probs_test = defaultdict(float)
    with open(test_model, 'r', encoding='utf-8') as f_test_model:
       for line in f_test_model:
           w, prob = line.strip().split('\t')
           probs_test[w] = float(prob)

    word_seg(test_input, probs_test, my_test_res)
    with open('acc.txt', 'a', encoding='utf-8') as f_acc:
        res = subprocess.run(
            f'diff -s {my_test_res} ../test/04-answer.txt'.split(),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding='utf-8'
        ).stdout

        f_acc.write(f'# diff between model results and ground-truth answers\n{res}\n')
        res_acc = subprocess.run(
            f'perl ../script/gradews.pl {test_answer} {my_test_res}'.split(),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding='utf-8'
        ).stdout
        f_acc.write(f'# acc for valuation\n{res_acc}\n')









