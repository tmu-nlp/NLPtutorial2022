'''
train_file = '../data/wiki-en-train.norm_pos'
model_res = 'model_results.txt'
pred_file = '../data/wiki-en-test.norm'

use script to test model results

pos tag prediction task: HMM
transition: pos_(i-1) -> pos_i
emit: pos_i - > word
'''

from collections import defaultdict
import math
import subprocess


# p8: make model file: compute probs for transition and emission
def train_hmm(train_file, model_file):
    with open(train_file, 'r', encoding='utf-8') as f_train, \
        open(model_file, 'w', encoding='utf-8') as f_model:
        emit = defaultdict(lambda: 0)
        transition = defaultdict(lambda: 0)
        context = defaultdict(lambda: 0)

        for line in f_train:
            previous = '<s>'
            context[previous] += 1
            wordtags = line.strip().split()
            for wordtag in wordtags:
                w, t = wordtag.split('_')
                transition[f'{previous} {t}'] += 1
                context[t] += 1
                emit[f'{t} {w}'] += 1
                previous = t
            transition[f'{previous} </s>'] += 1

        # 遷移確率を出力
        for k, v in transition.items():
            previous, w = k.split()
            f_model.write(f'T {k} {v/context[previous]}\n')
        # 生成確率を出力
        for k, v in emit.items():
            previous, w = k.split()
            f_model.write(f'E {k} {v/context[previous]}\n')


# using model output to predict pos of test_file
def predict_pos(model_file, pred_file, res_file):
    # p18
    transition = defaultdict(lambda: 0)
    emission = defaultdict(lambda: 0)
    possible_tags = {}
    with open(model_file, 'r' , encoding='utf-8') as f_model:
        for line in f_model:
            ty, ctxt, w, prob = line.split()
            possible_tags[ctxt] = 1
            if ty == 'T':
                transition[f'{ctxt} {w}'] = float(prob)   # P_T
            else:
                emission[f'{ctxt} {w}'] = float(prob)


    # p19: Viterbi algorithm
    with open(pred_file, 'r', encoding='utf-8') as f_pred, \
        open(res_file, 'w', encoding='utf-8') as f_res:
        for line in f_pred:
            # p19: forward step
            lambda_1, N = 0.95, 1000000
            words = line.split()
            words.append('</s>')
            l = len(words)
            best_score, best_edge = {}, {}
            best_score['0 <s>'] = 0
            best_edge['0 <s>'] = None
            for i in range(l):
                for prev in possible_tags.keys():
                    for next in possible_tags.keys():
                        if f'{i} {prev}' in best_score and f'{prev} {next}' in transition:
                            score = best_score[f'{i} {prev}']\
                                    - math.log(transition[f'{prev} {next}'], 2)\
                                    -math.log(lambda_1*emission[f'{next} {words[i]}'] + (1-lambda_1)/N, 2)
                            if f'{i+1} {next}' not in best_score or best_score[f'{i+1} {next}'] > score:
                                best_score[f'{i+1} {next}'] = score
                                best_edge[f'{i+1} {next}'] = f'{i} {prev}'

            # </s>に対して同じ操作
            best_score[f'{l+1} </s>'] = float('inf')
            for prev in possible_tags.keys():
                if f'{l} {prev}'in best_score and f'{prev} </s>' in transition:
                    score = best_score[f'{l} {prev}'] \
                            - math.log(transition[f'{prev} </s>'], 2)
                    if f'{l+1} {next}' not in best_score or best_score[f'{l+1} </s>'] > score:
                        best_score[f'{l+1} </s>'] = score
                        best_edge[f'{l+1} </s>'] = f'{l} {prev}'

            # p20: backward step
            tags = []
            next_edge = best_edge[f'{l+1} </s>']
            while next_edge != f'0 <s>':
                position, tag = next_edge.split()
                tags.append(tag)
                next_edge = best_edge[next_edge]
            tags.reverse()
            f_res.write(' '.join(tags[:-1]) + '\n')


if __name__ == '__main__':
    train_file = '../data/wiki-en-train.norm_pos'
    model_res = 'model_results.txt'
    pred_file = '../data/wiki-en-test.norm'
    pred_gt = '../data/wiki-en-test.pos'
    pos_res = 'pos_results.txt'
    res_acc = 'acc.txt'


    train_hmm(train_file, model_res)
    predict_pos(model_res, pred_file, pos_res)
    # write pos acc into a file
    with open(res_acc, 'w', encoding='utf-8') as f_acc:
        res = subprocess.run(
            f'perl ../script/gradepos.pl {pred_gt} {pos_res}'.split(),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding='utf-8'
        ).stdout
        f_acc.write(f'# pos accuracy\n{res}')













