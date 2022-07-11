from collections import defaultdict, deque
from tqdm import tqdm


class ShiftReduce:
    def __init__(self, train_file, test_file, out_file, epochs):
        self.feats = defaultdict(lambda: 0)  # 特徴量
        self.w = {}  # 重みが入った辞書
        self.w["SHIFT"] = defaultdict(lambda: 0)  # SHIFTになる重み
        self.w["R RIGHT"] = defaultdict(lambda: 0)  # R LIGHTになる重み
        self.w["R LEFT"] = defaultdict(lambda: 0)  # R LEFTになる重み
        self.train_file = train_file
        self.test_file = test_file
        self.out_file = out_file
        self.data = []
        self.epochs = epochs  # エポック数

    def makefeats(self, stack, queue):
        self.feats = defaultdict(lambda: 0)
        if len(stack) > 0 and len(queue) > 0:  # stackとqueueがどちらも空でない場合
            self.feats[f'W-1{stack[-1][1]}, W-0{queue[0][1]}'] += 1
            self.feats[f'W-1{stack[-1][1]}, P-0{queue[0][1]}'] += 1
            self.feats[f'P-1{stack[-1][2]}, W-0{queue[0][1]}'] += 1
            self.feats[f'P-1{stack[-1][2]}, P-0{queue[0][2]}'] += 1
        if len(stack) > 1:  # stackに2項以上積まれている場合
            self.feats[f'W-2{stack[-2][1]}, W-1{stack[-1][1]}'] += 1
            self.feats[f'W-2{stack[-2][1]}, P-1{stack[-1][2]}'] += 1
            self.feats[f'P-2{stack[-2][2]}, W-1{stack[-1][1]}'] += 1
            self.feats[f'P-2{stack[-2][2]}, P-1{stack[-1][2]}'] += 1

    def calculate_score(self):
        s_r = 0
        s_l = 0
        s_s = 0
        for key, value in self.feats.items():
            s_r += self.w['R RIGHT'][key] * value  # reduce right のスコア
            s_l += self.w['R LEFT'][key] * value  # reduce left のスコア
            s_s += self.w["SHIFT"][key] * value  # shift のスコア
        return s_r, s_l, s_s

    def calculate_correct(self, stack, heads, unproc):
        if len(stack) < 2:  # stackに一個しかなかったらreduceはありえないためshift
            correct = 'SHIFT'
        elif heads[stack[-1][0]] == stack[-2][0] and unproc[stack[-1][0]] == 0:
            correct = "R RIGHT"
        elif heads[stack[-2][0]] == stack[-1][0] and unproc[stack[-2][0]] == 0:
            correct = "R LEFT"
        else:
            correct = "SHIFT"
        return correct

    def calculate_ans(self, s_r, s_l, s_s, queue):
        if s_s >= s_r and s_s >= s_l and len(queue) > 0:
            ans = "SHIFT"
        elif s_l > s_r:
            ans = "R LEFT"
        else:
            ans = "R RIGHT"
        return ans

    def update_weight(self, ans, correct):
        for key, value in self.feats.items():
            self.w[ans][key] -= value
            self.w[correct][key] += value

    def shift_reduce_train(self, queue, heads):
        unproc = []
        stack = [(0, "ROOT", "ROOT")]
        for i in range(len(heads)):
            unproc.append(heads.count(i))
        while len(queue) > 0 or len(stack) > 1:
            self.makefeats(stack, queue)
            s_r, s_l, s_s = self.calculate_score()
            ans = self.calculate_ans(s_r, s_l, s_s, stack)
            correct = self.calculate_correct(stack, heads, unproc)
            if ans != correct:
                self.update_weight(ans, correct)
            if len(stack) < 2:
                stack.append(queue.popleft())
            elif heads[stack[-1][0]] == stack[-2][0] and unproc[stack[-1][0]] == 0:
                unproc[stack[-2][0]] -= 1
                del stack[-1]
            elif heads[stack[-2][0]] == stack[-1][0] and unproc[stack[-2][0]] == 0:
                unproc[stack[-1][0]] -= 1
                del stack[-2]
            else:
                stack.append(queue.popleft())

    def shift_reduce(self, queue):
        heads = [-1] * (len(queue)+1)
        stack = [(0, 'ROOT', 'ROOT')]
        while len(queue) > 0 or len(stack) > 1:
            self.makefeats(stack, queue)
            s_r, s_l, s_s = self.calculate_score()
            if len(stack) < 2 or (s_s >= s_l and s_s >= s_r and len(queue) > 0):
                stack.append(queue.popleft())
            elif s_l >= s_r:
                heads[stack[-2][0]] = stack[-1][0]
                del stack[-2]
            else:
                heads[stack[-1][0]] = stack[-2][0]
                del stack[-1]
        return heads

    def load_mst(self, file_name):
        queue = deque()
        heads = [-1]
        data = []
        with open(file_name, "r") as file:
            for line in file:
                if line == "\n":
                    data.append((queue, heads))
                    queue = deque()
                    heads = [-1]
                else:
                    id, word, _, pos, _, _, head, _ = line.strip().split('\t')
                    queue.append((int(id), word, pos))
                    heads.append(int(head))
            return data

    def train_sr(self):
        for _ in tqdm(range(self.epochs)):
            data = self.load_mst(self.train_file)
            for queue, heads in data:
                self.shift_reduce_train(queue, heads)

    def test_sr(self):
        with open(self.out_file, "w") as out:
            data = self.load_mst(self.test_file)
            for queue, _ in data:
                heads = self.shift_reduce(queue)
                for head in heads[1:]:
                    res = '_\t' * 6 + f'{head}\t_'
                    out.write(f'{res}\n')


if __name__ == "__main__":
    train_file = "mstparser-en-train.dep"
    test_file = "mstparser-en-test.dep"
    out_file = "tutorial11.txt"
    sr = ShiftReduce(train_file, test_file, out_file, 10)
    sr.train_sr()
    sr.test_sr()

"""
64.949343% (3013/4639)
"""