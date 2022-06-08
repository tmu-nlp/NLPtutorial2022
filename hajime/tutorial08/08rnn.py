from collections import defaultdict
import numpy as np
from tqdm import tqdm


class RNN():
    def __init__(self):
        self.x_ids = defaultdict(lambda: len(self.x_ids))
        self.y_ids = defaultdict(lambda: len(self.y_ids))
        self.train_data = []
        self.feat_lab = []
        self.net = []
        self.diff_net = []
        self.node = 64
        self.lam = 0.01

    def create_one_hot(self, id, size):
        vec = np.zeros(size)
        vec[id] = 1
        return vec

    def make_features(self, train_file):
        with open(train_file, 'r') as f:
            self.train_data = []
            for line in f:
                words = []
                poses = []
                line_list = line.strip().split()
                for word_pos in line_list:
                    word, pos = word_pos.split("_")
                    word = word.lower()
                    self.x_ids[word]
                    self.y_ids[pos]
                    words.append(word)
                    poses.append(pos)
                self.train_data.append([words, poses])

        for words, poses in self.train_data:
            word_vec = []
            pos_vec = []
            for word in words:
                word_vec.append(self.create_one_hot(
                    self.x_ids[word], len(self.x_ids)))
            for pos in poses:
                pos_vec.append(self.y_ids[pos])
            self.feat_lab.append([np.array(word_vec), np.array(pos_vec)])

    def rnn_init(self):
        W_rx = (np.random.rand(self.node, len(self.x_ids)) - 0.5)/5
        W_rh = ((np.random.rand(self.node, self.node) - 0.5))/5
        b_r = (np.zeros(self.node))
        W_oh = (np.random.rand(len(self.y_ids), self.node) - 0.5)/5
        b_o = (np.zeros(len(self.y_ids)))
        self.net = [W_rx, W_rh, b_r, W_oh, b_o]

    def rnn_diff_init(self):
        W_rx, W_rh, b_r, W_oh, b_o = self.net
        diff_W_rx = np.zeros(W_rx.shape)
        diff_W_rh = np.zeros(W_rh.shape)
        diff_b_r = np.zeros(b_r.shape)
        diff_W_oh = np.zeros(W_oh.shape)
        diff_b_o = np.zeros(b_o.shape)
        self.diff_net = [diff_W_rx, diff_W_rh, diff_b_r, diff_W_oh, diff_b_o]

    def forward_rnn(self, x):
        W_rx, W_rh, b_r, W_oh, b_o = self.net
        h = [0] * len(x)
        p = [0] * len(x)
        y = [0] * len(x)
        for t in range(len(x)):
            if t > 0:
                h[t] = np.tanh(np.dot(W_rx, x[t]) + np.dot(W_rh, h[t-1]) + b_r)
            else:
                h[t] = np.tanh(np.dot(W_rx, x[t]) + b_r)
            p[t] = np.tanh(np.dot(W_oh, h[t])+b_o)
            y[t] = np.argmax(p[t])
        return np.array(h), np.array(p), np.array(y)

    def gradient_rnn(self, x, h, p, y_d):
        W_rx, W_rh, b_r, W_oh, b_o = self.net
        self.rnn_diff_init()
        diff_W_rx, diff_W_rh, diff_b_r, diff_W_oh, diff_b_o = self.diff_net
        delta_r_d = np.zeros(len(b_r))
        for t in range(len(x)-1, -1, -1):
            p_d = self.create_one_hot(y_d[t], len(self.y_ids))
            delta_o_d = p_d - p[t]
            diff_W_oh += np.outer(delta_o_d, h[t])
            diff_b_o += delta_o_d
            delta_r = np.dot(delta_r_d, W_rh) + np.dot(delta_o_d, W_oh)
            delta_r_d = delta_r * (1 - h[t]**2)
            diff_W_rx += np.outer(delta_r_d, x[t])
            diff_b_r += delta_r_d
            if t != 0:
                diff_W_rh += np.outer(delta_r_d, h[t-1])

    def update_weights(self):
        for i in range(5):
            self.net[i] += self.lam * self.diff_net[i]

    def rnn_learn(self, train_file, iter):
        self.make_features(train_file)
        self.rnn_init()

        for _ in tqdm(range(iter)):
            for x, y_d in self.feat_lab:
                h, p, y = self.forward_rnn(x)
                self.gradient_rnn(x, h, p, y_d)
                self.update_weights()

    def rnn_test(self, test_file):
        with open(test_file, 'r') as t_file, open("my_answer_08", 'w') as a_file:
            for line in t_file:
                words = line.strip().split()
                vector = []
                x_ids_num = len(self.x_ids)
                for word in words:
                    if word in self.x_ids.keys():
                        vector.append(self.create_one_hot(
                            self.x_ids[word], x_ids_num))
                    else:
                        vector.append(np.zeros(x_ids_num))
                h, p, y = self.forward_rnn(vector)
                ans_pos = []
                id2pos = {self.y_ids[k]: k for k in self.y_ids}
                for pred in y:
                    ans_pos.append(str(id2pos[pred]))
                a_file.write(" ".join(ans_pos)+"\n")


if __name__ == "__main__":
    # train_file = "test/05-train-input.txt"
    # test_file = "test/05-test-input.txt"
    # iter = 10

    # rnn_1 = RNN()
    # rnn_1.rnn_learn(train_file, iter)
    # rnn_1.rnn_test(test_file)

    train_file = "data/wiki-en-train.norm_pos"
    test_file = "data/wiki-en-test.norm"
    iter = 10
    rnn_2 = RNN()
    rnn_2.rnn_learn(train_file, iter)
    rnn_2.rnn_test(test_file)
