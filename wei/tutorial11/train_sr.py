'''
学習 data/mstparser­en­train.dep
実行 data/mstparser­en­test.dep
評価 精度を測る script/grade­dep.py
'''

from collections import defaultdict, deque
from tqdm import tqdm
import subprocess


# p13
def make_feats(stack, queue):
    feats = defaultdict(lambda: 0)
    if len(stack) > 0 and len(queue) > 0:
        w_0 = queue[0][1]
        w_1 = stack[-1][1]
        p_0 = queue[0][2]
        p_1 = stack[-1][2]
        feats[f'W-1{w_1}, W-0{w_0}'] += 1
        feats[f'W-1{w_1}, P-0{p_0}'] += 1
        feats[f'P-1{p_1}, W-0{w_0}'] += 1
        feats[f'P-1{p_1}, P-0{p_0}'] += 1
    if len(stack) > 1:
        w_1 = stack[-1][1]
        w_2 = stack[-2][1]
        p_1 = stack[-1][2]
        p_2 = stack[-2][2]
        feats[f'W-2{w_2}, W-1{w_1}'] += 1
        feats[f'W-2{w_2}, P-1{p_1}'] += 1
        feats[f'P-2{p_2}, W-1{w_1}'] += 1
        feats[f'P-2{p_2}, P-1{p_1}'] += 1
    return feats


# p12
def cal_score(feats, w):
    s_r, s_l, s_s = 0.0, 0.0, 0.0
    for k, value in feats.items():
        s_r += w['right'][k] * value
        s_l += w['left'][k] * value
        s_s += w['shift'][k] * value
    return s_r, s_l, s_s


def cal_pred(s_r, s_l, s_s, queue):
    if s_s >= s_l and s_s >= s_r and len(queue) > 0:
        ans = 'shift'
    elif s_l > s_s and s_l >= s_r:
        ans = 'left'
    else:
        ans = 'right'
    return ans


def cal_correct(stack, heads, unproc):
    if len(stack) < 2:  # 1単語をキューからスタックへ移動
        correct = 'shift'
    elif heads[stack[-1][0]] == stack[-2][0] and unproc[stack[-1][0]] == 0:
        # 左が右の親、且つ右に未処理の子がない
        correct = 'right'
    elif heads[stack[-2][0]] == stack[-1][0] and unproc[stack[-2][0]] == 0:
        # 右が左の親、且つ右に未処理の子供がいない
        correct = 'left'
    else:
        correct = 'shift'
    return correct


# p17-19
def train_shift_reduce(queue, heads, w):
    stack = [(0, 'ROOT', 'ROOT')]
    unproc = []
    for i in range(len(heads)):
        unproc.append(heads.count(i))
    while len(queue) > 0 or len(stack) > 1:
        feats = make_feats(stack, queue)
        s_r, s_l, s_s = cal_score(feats, w)
        correct = cal_correct(stack, heads, unproc)
        ans = cal_pred(s_r, s_l, s_s, queue)
        if ans != correct:
            update_weights(w, feats, ans, correct)
        if correct == 'shift':
            stack.append(queue.popleft())
        elif correct == 'left':
            unproc[stack[-1][0]] -= 1
            del stack[-2]
        elif correct == 'right':
            unproc[stack[-2][0]] -= 1
            del stack[-1]


def update_weights(w, feats, ans, correct):
    for k,v in feats.items():
        w[ans][k] -= v
        w[correct][k] += v


# p15
def shift_reduce(queue, w):
    stack = [(0, 'ROOT', 'ROOT')]
    heads = [-1] * (len(queue) + 1)
    while len(queue) > 0 or len(stack) > 1:
        feats = make_feats(stack, queue)
        s_r, s_l, s_s = cal_score(feats, w)
        if s_s >= s_l and s_s >= s_r and len(queue) > 0 or len(stack)<2:
            stack.append(queue.pop())    # shift
        elif s_l > s_s and s_l >= s_r:
            heads[stack[-2][0]] = stack[-1][0]    # reduce left
            stack.pop(-2)
        else:
            heads[stack[-1][0]] = stack[-2][0]    # reduce right
            stack.pop(-1)
    return heads


def load_mst(file_path):
    queue, heads = deque(), [-1]
    data = []
    with open(file_path, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            line = line.rstrip()
            if line:
                id, word, _, pos, _, _, head, _ = line.split('\t')
                queue.append((int(id), word, pos))
                heads.append(int(head))

            else:
                data.append((queue, heads))
                queue, heads = deque(), [-1]
        return data

def train_sr(train_file, w, epochs=50):

    data = load_mst(train_file)
    for _ in tqdm(range(epochs)):
        for queue, heads in data:
            train_shift_reduce(queue, heads, w)



def test_sr(test_file, out_file , w):
    with open(out_file, 'w', encoding='utf-8') as f_out:
        data = load_mst(test_file)
        for queue, _ in data:
            heads = shift_reduce(queue, w)
            for head in heads[1:]:
                res = '_\t' * 6 + f'{head}\t_'
                f_out.write(f'{res}\n')
            print(file=f_out)


if __name__ == '__main__':
    train_file = '../data/mstparser-en-train.dep'
    test_file = '../data/mstparser-en-test.dep'
    results = 'results.txt'
    w = {}
    w['right'] = defaultdict(lambda: 0)
    w['left'] = defaultdict(lambda: 0)
    w['shift'] = defaultdict(lambda: 0)

    train_sr(train_file, w)
    test_sr(test_file, results, w)
    res_acc = subprocess.run(
        f'python ../script/grade-dep.py {test_file} {results}'.split(),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding='utf-8'
    ).stdout
    print(res_acc)




















