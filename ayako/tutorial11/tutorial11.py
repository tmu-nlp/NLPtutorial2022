from tqdm import tqdm
from collections import defaultdict

SHIFT, LEFT, RIGHT = range(3) #0, 1, 2

def make_features(stack, queue):
    """キューとスタックに基づいて素性を計算"""
    features = defaultdict(int)
    if len(stack) > 0 and len(queue) > 0:
        # ref:資料p.13
        # (id, 単語, 品詞)
        # -2->最後から2番，-1->最後，0->最初
        # stackの一番後ろとqueueの一番前の素性を数える
        features[f'W-1 {stack[-1][1]} W-0 {queue[0][1]}'] = 1
        features[f'W-1 {stack[-1][1]} P-0 {queue[0][2]}'] = 1
        features[f'P-1 {stack[-1][2]} W-0 {queue[0][1]}'] = 1
        features[f'P-1 {stack[-1][2]} P-0 {queue[0][2]}'] = 1
    if len(stack) > 1:
        #stackの後ろから2番目と1番後ろの素性を数える
        features[f'W-2 {stack[-2][1]} W-1 {stack[-1][1]}'] = 1
        features[f'W-2 {stack[-2][1]} P-1 {stack[-1][2]}'] = 1
        features[f'P-2 {stack[-2][2]} W-1 {stack[-1][1]}'] = 1
        features[f'P-2 {stack[-2][2]} P-1 {stack[-1][2]}'] = 1
    return features

def calc_score(w, feats):
    """素性と重みを掛け合わせてスコアを計算"""
    #ref:資料p.15
    #score = [shift, left, right]
    score = [0, 0, 0]
    for i in range(3):
        score[i] = sum(w[i][key] * value for key, value in feats.items())# key：W-2_word2_P-1_word1
    return score

def shift_reduce_train(queue, heads, w):
    #ref:資料p.17-19
    stack = [(0, 'ROOT', 'ROOT')]#特別なROOT記号のみ格納，stack[idx][0]：idx番目の単語のid
    unproc = [heads.count(i) for i in range(len(heads))]#各単語の未処理の子の数
    while len(queue) > 0 or len(stack) > 1:
        feats = make_features(stack, queue)
        s = calc_score(w, feats)

        #SHIFT，LEFT，RIGHTのどれかスコアが一番高いものを返す
        if s[SHIFT] >= s[LEFT] and s[SHIFT] >= s[RIGHT] and len(queue) > 0 or len(stack) < 2:
            ans = SHIFT
        elif s[LEFT] > s[SHIFT] and s[LEFT] >= s[RIGHT]:
            ans = LEFT
        else:
            ans = RIGHT

        if len(stack) < 2:
            corr = SHIFT
            stack.append(queue.pop(0))
        elif heads[stack[-1][0]] == stack[-2][0] and unproc[stack[-1][0]] == 0:
            # 左が右の親 and 左に未処理の子供がいない
            corr = RIGHT
            unproc[stack[-2][0]] -= 1
            stack.pop(-1)
        elif heads[stack[-2][0]] == stack[-1][0] and unproc[stack[-2][0]] == 0:
            # 右が左の親 and 右に未処理の子供がいない
            corr = LEFT
            unproc[stack[-1][0]] -= 1
            stack.pop(-2)
        else:
            corr = SHIFT
            stack.append(queue.pop(0))
        if ans != corr:
            update_weights(w, feats, ans, corr)

def shift_reduce_test(queue, w):
    #ref：資料p.17-19
    stack = [(0, 'ROOT', 'ROOT')]
    heads = [-1] * (len(queue) + 1)
    while len(queue) > 0 or len(stack) > 1:
        feats = make_features(stack, queue)
        s = calc_score(w, feats)
        if s[SHIFT] >= s[LEFT] and s[SHIFT] >= s[RIGHT] and len(queue) > 0 \
                or len(stack) < 2:
            stack.append(queue.pop(0))          # shift
        elif s[LEFT] > s[SHIFT] and s[LEFT] >= s[RIGHT]:
            heads[stack[-2][0]] = stack[-1][0]  # reduce 左
            stack.pop(-2)
        else:
            heads[stack[-1][0]] = stack[-2][0]  # reduce 右
            stack.pop(-1)
    return heads

def update_weights(w, feats, ans, corr):
    """重みを更新"""
    for key, value in feats.items():
        w[ans][key] -= value
        w[corr][key] += value

def load_data(path):
    queue, heads = [], [-1]
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip()
            if line: #空白行じゃないとき
                #ID 単語 原型 品詞 品詞2 拡張 親 ラベル
                id_, word, _, pos, _, _, head, _ = line.split('\t')
                queue += [(int(id_), word, pos)]
                heads += [int(head)]
            else: #空白行=文の終わりの時
                yield queue, heads #yield:途中で値を返すことができる
                queue = []
                heads = [-1]

def train_sr(train_path, epochs):
    for _ in tqdm(range(epochs)):
        for queue, heads in load_data(train_path):
            shift_reduce_train(queue, heads, w)

def test_sr(test_path, out_path):
    with open(out_path, 'w') as f_out:
        for queue, _ in load_data(test_path):
            heads = shift_reduce_test(queue, w)
            for head in heads[1:]:
                res = '_\t' * 6 + f'{head}\t_'
                print(res, file=f_out)
            print(file=f_out)

if __name__ == '__main__':
    path = "../../../nlptutorial/"
    train_path = path + "data/mstparser-en-train.dep"
    test_path = path + "data/mstparser-en-test.dep"
    out_path = "output.txt"
    epochs = 5

    w = [defaultdict(int) for _ in range(3)]
    
    train_sr(train_path, epochs)
    test_sr(test_path, out_path)

"""
epochs = 1
62.987713% (2922/4639)

epochs = 2
65.531365% (3040/4639)

epochs = 5
68.355249% (3171/4639)

epochs = 10
67.536107% (3133/4639)

epochs = 100
67.902565% (3150/4639)
"""