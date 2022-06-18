from tqdm import tqdm
import time
from collections import defaultdict
import argparse
import numpy as np


class RNN:
    def __init__(self, hidden_node=10, hidden_layer=1, lr=0.01):
        self.hidden_node = hidden_node
        self.hidden_layer = hidden_layer
        self.lr = lr
        self.word_ids = defaultdict(lambda: len(self.word_ids))
        self.pos_ids = defaultdict(lambda: len(self.pos_ids))
        self.feat_lab = []

    def init_net(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f_in:
            data_train = f_in.readlines()
            # initialize ids and make one-hot vecs
        for line in data_train:
            self.make_ids(line)
        for sent in data_train:
            sent_feat = self.create_feats(sent)
            self.feat_lab.append(sent_feat)

        # initialize net paras,-0.1~0.1の間の一様分布で初期化
        self.w_rx = np.random.rand(self.hidden_node, len(self.word_ids))/5 - 0.1
        self.w_rh = np.random.rand(self.hidden_node, self.hidden_node)/5 - 0.1
        self.b_r = np.random.rand(self.hidden_node)/5 - 0.1
        self.w_oh = np.random.rand(len(self.pos_ids), self.hidden_node)/5 - 0.1
        self.b_o = np.random.rand(len(self.pos_ids))/5 - 0.1

    def make_ids(self, X):
        words = X.strip().split()
        for word in words:
            x, y = word.split('_')
            x.lower()
            self.word_ids[x]    # {word:id, ...}
            self.pos_ids[y]     # {pos:id, ...}
        return self.word_ids, self.pos_ids

    def create_feats(self, X):
        feats = []
        line = X.strip().split()
        for word_pos in line:
            x, y = word_pos.split('_')
            x.lower()
            x_vec = self.create_one_hot(self.word_ids[x], len(self.word_ids))
            y_vec = self.create_one_hot(self.pos_ids[y], len(self.pos_ids))
            feats.append([x_vec, y_vec])
        return feats


    def create_one_hot(self, id, size):
        vec = np.zeros(size)
        vec[id] = 1
        return vec

    # p32
    def train_rnn(self, train_data, lr, iter):
        self.init_net(train_data)
        for _ in tqdm(range(iter)):
            for sent in self.feat_lab:     # y_dはpos label
                h, p, _ = self.forward_rnn(sent)
                para_gradient = self.gradient_rnn(sent, p, h)
                net = [self.w_rx, self.w_rh, self.b_r, self.w_oh, self.b_o]
                self.update_weights(net, para_gradient, lr)

                with open('weights.txt', 'w', encoding='utf-8') as wf,\
                    open('ids.txt', 'w', encoding='utf-8') as idf:
                    weights = ['w_rx', 'w_rh', 'b_r', 'w_oh', 'b_o']
                    for x in zip(weights, net):
                        wf.write(f'{x[0]}\n{x[1]}\n')
                    idf.write('word_ids\n')
                    for k,v in self.word_ids.items():
                        idf.write(f'{v}\t{k}\n')
                    idf.write('pos_ids\n')
                    for k,v in self.pos_ids.items():
                        idf.write(f'{v}\t{k}\n')

    #p16
    def forward_rnn(self, sent):
        h = []
        p = []
        y = []
        for t in range(len(sent)):
            x, _ = sent[t]
            if t > 0:
                h.append(np.tanh(np.dot(self.w_rx, x) + np.dot(self.w_rh, h[t-1]) + self.b_r))
            else:
                h.append(np.tanh(np.dot(self.w_rx, x) + self.b_r))

            p.append(np.tanh(np.dot(self.w_oh, h[t]) + self.b_o))
            #p.append(self.softmax(np.dot(self.w_oh, h[t]) + self.b_o))
            y.append(self.find_max(p[t]))
        return h, p, y

    # def softmax(self, x):
    #     u = np.sum(np.exp(x))
    #     return np.exp(x)/u

    def find_max(self, p):
        y = 0
        for i in range(len(p)):
            if p[i] > p[y]:
                y = i
        return y            # return index with maximum value in array p

    # p30
    def gradient_rnn(self, sent, p, h):         # y_dは正解
        # 勾配重みを0で初期化
        dw_rx = np.zeros((self.hidden_node, len(self.word_ids)))
        dw_rh = np.zeros((self.hidden_node, self.hidden_node))
        db_r = np.zeros(self.hidden_node)
        dw_oh = np.zeros((len(self.pos_ids), self.hidden_node))
        db_o = np.zeros(len(self.pos_ids))
        delta_r_d = np.zeros(self.hidden_node)

        for t in reversed(range(len(sent))):
            x, y = sent[t]
            delta_out_d = y - p[t]
            dw_oh += np.outer(delta_out_d, h[t])
            db_o += delta_out_d

            delta_r = np.dot(delta_r_d, self.w_rh) + np.dot(delta_out_d, self.w_oh)
            delta_r_d = delta_r * (1-h[t]**2)
            dw_rx += np.outer(delta_r_d, x)
            db_r += delta_r_d
            if t != 0:
                dw_rh += np.outer(delta_r_d, h[t-1])
            para_grad = dw_rx, dw_rh, db_r, dw_oh, db_o

        return para_grad

    def update_weights(self, net, para_grad, lr):
        for i in range(len(net)):
            net[i] += lr * para_grad[i]

        return net

    def create_pred_feats(self, line):
        phis = []
        words = line.strip().split(line)
        for word in words:
            if word in self.word_ids:
                phis.append(self.create_one_hot(self.word_ids[word], len(self.word_ids)))
            else:
                phis.append(np.zeros(len(self.word_ids)))

        return phis







    def predict_all(self, to_predict, results):
        with open(to_predict, 'r', encoding='utf-8') as f_pred, \
                open(results, 'w', encoding='utf-8') as f_res:
            data2pred = f_pred.readlines()

            for line in data2pred:
                sent = []
                phis = self.create_pred_feats(line)   # create features of pred_data
                h, p, y = self.predict_one(phis)
                for ans in y:
                    for v, id in self.pos_ids.items():
                        if id == ans:
                            sent.append(f'{v}\t')
                f_res.write(f'{"".join(sent)}\n')

    def predict_one(self, sent):
        h = []
        p = []
        y = []
        for t in range(len(sent)):
            if t>0:
                h.append(np.tanh(np.dot(self.w_rx, sent[t]) + np.dot(self.w_rh, h[t-1]) + self.b_r))
            else:
                h.append(np.tanh(np.dot(self.w_rx, sent[t]) + self.b_r))
            p.append(np.tanh(np.dot(self.w_oh, h[t]) + self.b_o))
            y.append(self.find_max(p[t]))
        return h, p, y


if __name__ == '__main__':
    start = time.time()

    train_file = '../data/wiki-en-train.norm_pos'
    to_pred_file = '../data/wiki-en-test.norm'
    res = 'pred_reslults.txt'

    lr = 0.01
    iter = 20
    rnn =RNN()
    rnn.train_rnn(train_file, lr, iter)
    rnn.predict_all(to_pred_file, res)

    end = time.time()
    print(f'time used : {end-start}s')

# time used : 631.9778416156769s





