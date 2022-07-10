from collections import deque, defaultdict
from tqdm import tqdm


class ShiftReduce():
    def __init__(self):
        self.w_s = defaultdict(int)
        self.w_l = defaultdict(int)
        self.w_r = defaultdict(int)
        self.w = [self.w_s, self.w_l, self.w_r]

    def make_feats(self, stack, queue):
        features = defaultdict(int)
        if len(stack) > 0 and len(queue) > 0:
            features[f'W-1 {stack[-1][1]} W-0 {queue[0][1]}'] += 1
            features[f'W-1 {stack[-1][1]} P-0 {queue[0][2]}'] += 1
            features[f'P-1 {stack[-1][2]} W-0 {queue[0][1]}'] += 1
            features[f'P-1 {stack[-1][2]} P-0 {queue[0][2]}'] += 1
        if len(stack) > 1:
            features[f'W-2 {stack[-2][1]} W-1 {stack[-1][1]}'] += 1
            features[f'W-2 {stack[-2][1]} P-1 {stack[-1][2]}'] += 1
            features[f'P-2 {stack[-2][2]} W-1 {stack[-1][1]}'] += 1
            features[f'P-2 {stack[-2][2]} P-1 {stack[-1][2]}'] += 1
        return features

    def calc_score(self, feats):
        score = [0, 0, 0]
        for key, value in feats.items():
            for i in range(3):
                score[i] += self.w[i][key] * value
        return score

    def load_mst(self, input_file):
        with open(input_file, "r")as i_file:
            queue = []
            heads = [-1]
            for line in i_file:
                """ id	word	surface	pos1	pos2	extend	head	label """
                line = line.rstrip("\n")
                if line:
                    line = line.split("\t")
                    id_num, word, pos, head = line[0], line[2], line[3], line[6]
                    id_num = int(id_num)
                    head = int(head)
                    queue.append((id_num, word, pos))
                    heads.append(head)
                else:
                    yield queue, heads
                    queue = []
                    heads = [-1]

    def calc_ans(self, score, queue, stack):
        if (score[0] >= score[1] and score[0] >= score[2] and len(queue) > 0) or len(stack) < 2:
            ans = "shift"
        elif score[1] > score[0] and score[1] > score[2]:
            ans = "left"
        else:
            ans = "right"
        return ans

    def calc_corr(self, stack, queue, heads, unproc):
        if len(stack) < 2:
            corr = "shift"
            stack.append(queue.pop(0))
        elif heads[stack[-1][0]] == stack[-2][0] and unproc[stack[-1][0]] == 0:
            corr = "right"
            unproc[stack[-2][0]] -= 1
            stack.pop(-1)
        elif heads[stack[-2][0]] == stack[-1][0] and unproc[stack[-2][0]] == 0:
            corr = "left"
            unproc[stack[-1][0]] -= 1
            stack.pop(-2)
        else:
            corr = "shift"
            stack.append(queue.pop(0))
        return corr, stack, queue, unproc, heads

    def update_weights(self, feats, ans, corr):
        for key, value in feats.items():
            if ans == "shift":
                self.w[0][key] -= value
            elif ans == "left":
                self.w[1][key] -= value
            else:
                self.w[2][key] -= value

            if corr == "shift":
                self.w[0][key] += value
            elif corr == "left":
                self.w[1][key] += value
            else:
                self.w[2][key] += value

    def shift_reduce(self, queue):
        heads = [-1] * (len(queue)+1)
        stack = [(0, "ROOT", "ROOT")]
        while len(queue) > 0 or len(stack) > 1:
            feats = self.make_feats(stack, queue)
            score = self.calc_score(feats)
            if (score[0] >= score[1] and score[0] >= score[2] and len(queue) > 0) or len(stack) < 2:
                stack.append(queue.pop(0))
            elif score[1] >= score[2]:
                heads[stack[-2][0]] = stack[-1][0]
                stack.pop(-2)
            else:
                heads[stack[-1][0]] = stack[-2][0]
                stack.pop(-1)
        return heads

    def shift_reduce_train(self, queue, heads):
        unproc = []
        stack = [(0, "ROOT", "ROOT")]
        for i in range(len(heads)):
            unproc.append(heads.count(i))
        while len(queue) > 0 or len(stack) > 1:
            feats = self.make_feats(stack, queue)
            score = self.calc_score(feats)

            # ans = self.calc_ans(score, stack, queue)
            if (score[0] >= score[1] and score[0] >= score[2] and len(queue) > 0) or len(stack) < 2:
                ans = 'shift'
            elif score[1] > score[0] and score[1] > score[2]:
                ans = "left"
            else:
                ans = "right"

            # corr, stack, queue, unproc, heads = self.calc_corr(
            #     stack, queue, heads, unproc)
            if len(stack) < 2:
                corr = "shift"
                stack.append(queue.pop(0))
            elif heads[stack[-1][0]] == stack[-2][0] and unproc[stack[-1][0]] == 0:
                corr = "right"
                unproc[stack[-2][0]] -= 1
                stack.pop(-1)
            elif heads[stack[-2][0]] == stack[-1][0] and unproc[stack[-2][0]] == 0:
                corr = "left"
                unproc[stack[-1][0]] -= 1
                stack.pop(-2)
            else:
                corr = "shift"
                stack.append(queue.pop(0))

            if ans != corr:
                self.update_weights(feats, ans, corr)

    def train(self, train_file, epoch=5):
        for _ in tqdm(range(epoch)):
            for queue, heads in self.load_mst(train_file):
                self.shift_reduce_train(queue, heads)

    def test(self, test_file):
        with open("output/11_output.txt", "w") as o_file:
            for queue, _ in self.load_mst(test_file):
                heads_test = self.shift_reduce(queue)
                heads_test.pop(0)
                for i in range(len(heads_test)):
                    output = f"_\t_\t_\t_\t_\t_\t{heads_test[i]}\t_\n"
                    o_file.write(output)
                o_file.write("\n")


if __name__ == "__main__":
    train_file = "data/mstparser-en-train.dep"
    test_file = "data/mstparser-en-test.dep"
    dep = ShiftReduce()
    dep.train(train_file, 20)
    dep.test(test_file)

"""
epoch:20
63.246389% (2934/4639)
"""
