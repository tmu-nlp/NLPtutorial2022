'''
学習1回、隠れ層1つ、隠れ層のノード2つ
'''
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import time



class NeuralNet:
    # 隠れ層1つ、隠れ層のノード2つ
    def __init__(self, layer_num, node_num, learning_rate):
        # 変数の初期化
        self.layer = layer_num
        self.node = node_num
        self.lr = learning_rate
        self.ids = defaultdict(lambda: len(self.ids))
        # 创建空容器{}，在init_ids()时得到file vocab的ids，like {'UNI:apple':0, 'UNI:banana':1, ..., 'UNI:tomato':len(ids)-1}
        self.feat_lab = []
        # store hidden vector and gt label
        self.paras = []

    # uni-gram素性を抽出

    def init_ids(self, input_file):
        x_y_list = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                label, sentence = line.rstrip().split('\t')
                label = int(label)
                for word in sentence.split():
                    self.ids[f'UNI:{word}']
                    # create uni-gram ids of whole vocab(words present in file), first id is 0. len(ids) = vocab size
                x_y_list.append([sentence, label])

            self.init_net()
            for sent, lab in x_y_list:
                self.feat_lab.append([self.create_features(sent), int(lab)])
            # list_of_list: [[defaultdict obj, label],...]
            # e.g.:   [[defaultdict(<function <lambda> at >,
            # [[{id: counts,...},1],[{id: counts,...},-1]...]

    # w, bをランダムな値で初期化
    def init_net(self):
        # W_0 :2 nodes in hidden layer
        w_0 = np.random.rand(self.node, len(self.ids))/5 - 0.1  # -0.1~0.1の間の一様分布
        b_0 = np.random.rand(self.node)/5 - 0.1
        self.paras.append([w_0, b_0])

        # hidden_1から出力層直前の層までのw, b
        for _ in range(self.layer - 1):
            w = np.random.rand(self.node, self.node)/5 - 0.1
            b = np.random.rand(self.node)/5 - 0.1
            self.paras.append([w, b])

        # w, b of output layer
        w_o = np.random.rand(1, self.node)/5 - 0.1
        b_o = np.random.rand(1)/5 - 0/1
        self.paras.append([w_o, b_o])

    def create_features(self, x):
        phi = [0 for _ in range(len(self.ids))]
        words = x.split()
        for word in words:
            if f'UNI:{word}' in self.ids:
                phi[self.ids['UNI:' + word]] += 1
                # id: counts(counts=len(ids) + 出现次数, len(ids)并不是固定长度，当phi中没有并创建新key，将len(ids) + 1 )
        return phi       # like {id :counts of id_word, ....}

    # ニューラルネットワークの学習
    def train_nn(self, iter_num):
        for _ in tqdm(range(iter_num)):
            for phi_0, y_d in self.feat_lab:
                # feat_tab: [[defaultdict obj, label],...]
                phis = self.forward_nn(phi_0)                # sentence featuresを入れる
                print(type(phis))
                delta_d = self.backward_nn(phis, y_d)      # y_d は正解ラベル
                self.update_w(phis, delta_d)

    # 順伝播,2p6~p28
    def forward_nn(self, phi_0):
        # phi_0 = {id :counts of id_word, ....}
        phis = [0 for _ in range(len(self.ids)+1)]
        phis[0] = phi_0
        for i in range(len(self.paras)):
            w, b = self.paras[i]
            print(f'shape of w in forward is {w.shape}')
            # 前の層の値に基づいて値を計算、内積の計算結果を非線形化
            phis[i+1] = np.tanh(np.dot(w, phis[i]) + b)

        return phis

    # 逆伝播:出力層からエラーを後ろへ伝播 p29~p31
    def backward_nn(self, phis, y_d):
        J = len(self.paras)
        delta = [0 for _ in range(J)]
        delta.append(y_d - phis[J][0])        # 最終層からエラーの勾配を計算
        delta_d = [0 for _ in range(J+1)]  # gradient of error
        for i in reversed(range(J)):    # p31
            delta_d[i+1] = delta[i+1] * (1-phis[i+1]**2)  # p29
            w, b = self.paras[i]
            print(f'type of w_back is {type(w)}, type of delta_d[i+1] is {type(delta_d[i+1])}')
            delta[i] = np.dot(delta_d[i+1], w)
        print(f'length of delta_d is {len(delta_d)}')
        return delta_d


    # 重み更新p32: wの勾配は、次のdelta_dと前のphiの外積で求める
    def update_w(self, phis, delta_d):
        for i in range(len(self.paras)):
            w, b = self.paras[i]
            w += self.lr * np.outer(delta_d[i+1], phis[i])
            b += self.lr * delta_d[i+1]



    # ファイル全体の予測
    def predict_all(self,to_predict_file, pred_result):
        with open(to_predict_file, 'r', encoding='utf-8') as f1:
            to_pred = f1.readlines()
        with open(pred_result, 'w', encoding='utf-8') as f2:
            for line in to_pred:
                line = line.strip()
                phi_0 = self.create_features(line)
                y_pred = self.predict_one(phi_0)
                f2.write(f'{y_pred}\t{line}\n')

    # `一文の予測
    def predict_one(self, phi_0):
        phis = self.forward_nn(phi_0)
        score = phis[len(self.paras)]
        if score >= 0:
            return 1
        else:
            return -1

    def output_weights(self, out_f):
        with open(out_f, 'w', encoding='utf-8') as f:
            for i, (w,b) in enumerate(self.paras):
                f.write(f'layer {i}\n{w}\n{b}\n')

    def output_ids(self, out_ids):
        with open(out_ids, 'w', encoding='utf-8') as f:
            for word, id in self.ids.items():
                f.write(f'{word}\t{id}\n')




if __name__ == '__main__':
    start = time.time()

    train_file = '../data/titles-en-train.labeled'
    #train_file_pre = './preprossed_data.txt'
    id_output = 'ids.txt'
    weight_output = 'weights.txt'
    to_pred_file = '../data/titles-en-test.word'
    pred_result = 'pred_result.txt'
    # to_cal_accu = '../data/titles-en-test.labeled'

    nn = NeuralNet(1, 2, 0.01)
    nn.init_ids(train_file)
    nn.train_nn(1)
    nn.output_ids(id_output)
    nn.output_weights(weight_output)
    nn.predict_all(to_pred_file, pred_result)


    end = time.time()

    print(f'time used : {end-start}s')
    # python train_nn.py ../data/titles-en-train.labeled ids.txt weights.txt ../data/titles-en-test.word pred_result.txt