'''
λ_1=0.95，λ_UNK=0.05, V=1000000, W=0, H=0
test_file:data/wiki-en-test.word に対してエントロピーとカバレージを計算
'''

import math

# 評価と結果を表示
def test_model(test_file, model_out):
    # モデルを読み込む
    probs = {}
    with open(model_out, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            word, prob = line.split('\t')
            probs[word] = prob

    lamb, V, W, H, cnt_unk = 0.95, 1000000, 0, 0, 0
    with open(test_file, 'r', encoding='utf-8') as f_test:
        for line in f_test:
            words = line.strip().split()
            words.append('</s>')
            for word in words:
                W += 1
                P = (1-lamb)/V
                if word in probs:
                    P += lamb * float(probs[word])
                else:
                    cnt_unk += 1

                H += -math.log(P, 2)

    print(f'Entropy：{H/W}')
    print(f'Coverage: {(W-cnt_unk)/W}')


if __name__ == '__main__':
    model_file = 'model_file.txt'
    test_file = '../data/wiki-en-test.word'
    test_model(test_file, model_file)
'''
Entropy：10.527343216681667
Coverage: 0.895226024503591'''