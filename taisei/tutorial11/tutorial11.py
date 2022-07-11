from collections import defaultdict
import math 
from tqdm import tqdm

class SR:
    def __init__(self):
        self.stack = []
        self.weights = {}
        self.weights['shift'] = defaultdict(lambda: 0)
        self.weights['left'] = defaultdict(lambda: 0)
        self.weights['right'] = defaultdict(lambda: 0)


    def make_feats(self, queue):
        feats = defaultdict(lambda: 0)
        if len(self.stack) > 0 and len(queue) > 0:
            w_0, p_0 = queue[0][1], queue[0][2]
            w_1, p_1 = self.stack[-1][1], self.stack[-1][2]
            feats[f'W-1{w_1},W-0{w_0}'] += 1
            feats[f'W-1{w_1},P-0{p_0}'] += 1
            feats[f'P-1{p_1},W-0{w_0}'] += 1
            feats[f'P-1{p_1},P-0{p_0}'] += 1
        if len(self.stack) > 1:
            w_1, p_1 = self.stack[-1][1], self.stack[-1][2]
            w_2, p_2 = self.stack[-2][1], self.stack[-2][2]
            feats[f'W-2{w_2},W-1{w_1}'] += 1
            feats[f'W-2{w_2},P-1{p_1}'] += 1
            feats[f'P-2{p_2},W-1{w_1}'] += 1
            feats[f'P-2{p_2},P-1{p_1}'] += 1
        return feats


    def calculate_ans(self, feats, queue):
        shift, r_left, r_right = 0, 0, 0
        for name, value in feats.items():
            r_left += self.weights['right'][name] * value
            r_right += self.weights['left'][name] * value
            shift += self.weights['shift'][name] * value
        if shift >= r_left and shift >= r_right and len(queue) > 0:
            predict = 'shift'
        elif r_left >= r_right:
            predict = 'right'
        else:
            predict = 'left'
        return predict


    def calculate_correct(self, heads, unproc):
        if len(self.stack) < 2:
            correct = 'shift'
        elif heads[self.stack[-1][0]] == self.stack[-2][0] and unproc[self.stack[-1][0]] == 0:
            correct = 'right'
        elif heads[self.stack[-2][0]] == self.stack[-1][0] and unproc[self.stack[-2][0]] == 0:
            correct = 'left'
        else:
            correct = 'shift'
        return correct


    def shift_reduce_train(self, heads, queue):
        unproc = [] # 各単語の未処理の子どもの数
        self.stack = [(0, "ROOT", "ROOT")]
        for i in range(len(heads)):
            unproc.append(heads.count(i))
        while len(queue) > 0 or len(self.stack) > 1:
            feats = self.make_feats(queue)
            pred = self.calculate_ans(feats, queue)
            corr = self.calculate_correct(heads, unproc)
            if pred != corr:
                for name, value in feats.items():
                    self.weights[pred][name] -= value
                    self.weights[corr][name] += value
            if corr == 'shift':
                self.stack.append(queue.pop(0))
            elif corr == 'left':
                unproc[self.stack[-1][0]] -= 1
                del self.stack[-2]
            elif corr == 'right':
                unproc[self.stack[-2][0]] -= 1
                del self.stack[-1]


    def shift_reduce_test(self, queue):
        heads = [-1] * (len(queue) + 1)
        self.stack = [(0, 'ROOT', 'ROOT')]
        while len(queue) > 0 or len(self.stack) > 1:
            feats = self.make_feats(queue)
            ans = self.calculate_ans(feats, queue)
            if len(self.stack) < 2 or ans == 'shift':
                self.stack.append(queue.pop(0))
            elif ans == 'left':
                heads[self.stack[-2][0]] = self.stack[-1][0]
                del self.stack[-2]
            elif ans == 'right':
                heads[self.stack[-1][0]] = self.stack[-2][0]
                del self.stack[-1]
        return heads

if __name__ == '__main__':
    data = []
    queue = []
    heads = [-1]
    EPOCH = 50
    k = SR()
    with open('mstparser-en-train.dep', 'r') as trainfile:
        for line in trainfile:
            if line != '\n':
                id, surface, base, pos1, pos2, _, parent, label = line.strip().split('\t')
                queue.append((int(id), surface, pos1, pos2))
                heads.append(int(parent))
            else:
                data.append((queue, heads))
                queue = []
                heads = [-1]
    
    # training
    for _ in tqdm(range(EPOCH)):
        for queue, heads in data:
            k.shift_reduce_train(heads, queue)

    data = []
    queue = []
    ans_data = []
    ans_file_element = []
    with open('mstparser-en-test.dep', 'r') as testfile:
        for line in testfile:
            if line != '\n':
                id, surface, base, pos1, pos2, _, parent, label = line.strip().split('\t')
                queue.append((int(id), surface, pos1, pos2)) ##
                ans_file_element.append((id, surface, base, pos1, pos2, _, parent, label))
            else:
                data.append(queue)
                queue = []
                ans_data.append(ans_file_element)
                ans_file_element = []
    
    # test
    heads_list = []
    for queue in data:
        heads_list.append(k.shift_reduce_test(queue))
    with open('ans_file', 'w') as ansfile:
        for i, heads in enumerate(heads_list):
            for j, head in enumerate(heads):
                if j == 0: continue
                id, surface, base, pos1, pos2, _, parent, label = ans_data[i][j-1]
                ansfile.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(id, surface, base, pos1, pos2, _, head, label))
            ansfile.write('\n')

"""
python2 ../../../nlptutorial/script/grade-dep.py ans_file mstparser-en-test.dep
61.694331% (2862/4639)
"""