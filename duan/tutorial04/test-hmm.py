from collections import defaultdict
import math

transition = defaultdict(lambda:0) # 遷移確率を格納する
emission = defaultdict(lambda:0) # 生成確率を格納する
possible_tags = defaultdict(lambda:0) # タグを格納する
lam = 0.95; V = 1000000


model_file = open('./NLPtutorial2022/duan/tutorial04/model.txt')
# モデルを読み込む
for line in model_file:
    line = line.rstrip() # 改行コードを除去する
    type, context, word, prob = line.split(' ')
    possible_tags[context] = 1 # 可能なタグとして保存する
    if type == 'T':
        transition[f'{context} {word}'] = float(prob)
    else:
        emission[f'{context} {word}'] = float(prob)

test = open('./nlptutorial/data/wiki-en-test.norm')
output = open('./NLPtutorial2022/duan/tutorial04/my_answer.pos','w')

# 前向きステップ：各ノードへたどる確率の計算
for line in test:
    line = line.rstrip()
    words = line.split(' ')
    l = len(words)
    best_score = {}; best_edge = {}
    best_score['0 <s>'] = 0 # <s>から始まる
    best_edge['0 <s>'] = None

    for i in range(l):
        for prev in possible_tags.keys():
            for next in possible_tags.keys():
                if f'{i} {prev}' in best_score and f'{prev} {next}' in transition:
                    # HMM遷移確率
                    P_T = transition[f'{prev} {next}']
                    # HMM生成確率
                    P_E = lam * emission[f'{next} {words[i]}'] + (1-lam)/V
                    score = best_score[f'{i} {prev}'] - math.log2(P_T) - math.log2(P_E)
                    if f'{i+1} {next}' not in best_score or best_score[f'{i+1} {next}'] > score:
                        # ベストスコアを更新する
                        best_score[f'{i+1} {next}'] = score
                        best_edge[f'{i+1} {next}'] = f'{i} {prev}'

    # </s>に対して同じ操作をを行う
    for tag in possible_tags.keys():
        if  f'{l} {tag}' in best_score and f'{tag} </s>' in transition:
            # HMM遷移確率
            P_T = transition[f'{tag} </s>']
            score = best_score[f'{l} {tag}'] - math.log2(P_T)
            if f'{l+1} </s>' not in best_score or best_score[f'{l+1} </s>'] > score:
                best_score[f'{l+1} </s>'] = score
                best_edge[f'{l+1} </s>'] = f'{l} {tag}'

    # 後ろ向きステップ：パスの復元
    tags = []
    next_edge = best_edge[f'{l+1} </s>']
    while next_edge != '0 <s>':
        # エッジの品詞を出力に追加する
        position, tag = next_edge.split(' ')
        tags.append(tag)
        next_edge = best_edge[next_edge]
    tags.reverse()
    print(' '.join(tags), file=output)
