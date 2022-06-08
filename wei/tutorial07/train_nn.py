'''
学習1回、隠れ層1つ、隠れ層のノード2つ

'''
from collections import defaultdict
import string
import re
import pandas as pd
import numpy as np
import tqdm


class NN:
    # 隠れ層1つ、隠れ層のノード2つ
    def __init__(self, layer_num, node_num, out_num):
        # 変数の初期化
        self.layer = layer_num
        self.node = node_num
        self.out_num = out_num
        self.ids = defaultdict(lambda: len(self.ids))   # 素数の数、各特徴量のIDが入っている辞書
        self.feat_lab = []   # store hidden vector and gt label
        self.paras = []

    # 前処理
    def preprocessing(self, x):
        x = x.lower()
        trantab = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        x = x.translate(trantab)
        x = re.sub(r'[0-9]+', '0', x)

        return x

    def load_df(self, file):
        file = pd.read_table(file, names = ['label', 'sentence'])
        file['sentence'] = file['sentence'].map(lambda x: self.preprocessing(x))

        return file

    # uni-gram素性を抽出
    def create_features(self, x):
        phi = defaultdict(lambda :0)
        words = x.split()
        for word in words:
            if f'UNI:{word}' in self.ids:
                phi[self.ids['UNI:' + word]] += 1
        return phi       # something like {'UNI:a': 2, ..., 'UNI:d': 1})

    def store_features(self, file):
        x_y_list = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in file:
                label, sentence = line.rstrip().split('\t')
                label = int(label)
                for word in sentence.split():
                    self.ids['UNI:' + word]      # 各特徴量にIDを振り分け
                x_y_list.append([sentence, label])
        for sentence, label in x_y_list:
            self.feat_lab.append([self.create_features(sentence), int(label)])
            # list_of_list: [[defaultdict obj, label],...]
            # e.g.:   [[defaultdict(<function __main__.<lambda>()>,
            # {'UNI:a': 2, 'UNI:b': 2, 'UNI:c': 1, 'UNI:d': 1}),1],...]
        return x_y_list



    # ニューラルネットワークの学習
    def train_nn(self, iter_num, lr):
        for _ in tqdm(range(iter_num)):
            for phi_0, y in tqdm(self.feat_lab):
                # feat_tab: [[defaultdict obj, label],...]
                phi = self.forward_nn(phi_0)    #
                delta_grad = self.backward_nn(phi, y)
                self.update_w(phi, delta_grad, lr)

        return self.paras, self.ids

    # w, bをランダムな値で初期化
    def init_net(self):
        # W_0 :2 nodes in hidden layer
        w_0 = np.random.rand(self.node, len(self.ids))/5 - 0.1  # -0.1~0.1の間の一様分布
        b_0 = np.random.rand(self.node)/5 - 0.1
        self.paras.append((w_0, b_0))

        # hidd_1 から出力層直前の層までのw, b
        for _ in range(self.layer - 1):
            w = np.random.rand(self.node, self.node)/5 - 0.1
            b = np.random.rand(self.node)/5 - 0.1
            self.paras.append((w, b))

        # w, b of output layer
        w_o = np.random.rand(self.out_num, self.node)/5 - 0.1
        b_o = np.random.rand(self.out_num)/5 - 0/1
        self.paras.append((w_o, b_o))


    # 順伝播,2p6~p28
    def forward_nn(self, phi_0):
        # phi_0 = [defaultdict(<function __main__.<lambda>()>,
        #             # {'UNI:a': 2, 'UNI:b': 2, 'UNI:c': 1, 'UNI:d': 1}),1]
        phi = [0 for _ in range(len(self.ids)+1)]  # 各層の値
        phi[0] = phi_0
        for i in range(len(self.paras)):
            w, b = self.paras[i]
            # 前の層の値に基づいて値を計算、内積の計算結果を非線形化した
            phi[i+1] = np.tanh(np.dot(w, phi[i])+b)
        return phi

    # 逆伝播:出力層からエラーを後ろへ伝播 p29~p31
    def backward_nn(self, phi, y_pred):
        J = len(self.paras)
        delta = np.array([0 for _ in range(J)])
        delta.append(y_pred - phi[J])    # エラーの勾配
        delta_grad = np.array([0 for _ in range(J+1)])
        for i in reversed(range(J)):    # p31
            delta_grad[i+1] = delta[i+1] * (1-phi[i+1]**2)  # p29:back propagation of error
            w, b = self.paras[i]
            delta[i] = np.dot(delta_grad[i+1], w)
        return delta_grad


    # 重み更新: wの勾配は、次のdelta_gradと前のphiの外積で求める
    def update_w(self,phi, delta_pred, lr):
        for i in range(len(self.paras)):
            w, b = self.paras[i]
            w += lr * np.outer(delta_pred[i+1], phi[i])
            b += lr * delta_pred[i+1]



    # 予測
    def predict_all(self,to_predict_file):
        with open(to_predict_file, 'r' ,encoding='utf-8') as f:
            for line in f:
                x = line.strip()
                phi_0 = self.create_features(x)
                label = self.predict_one(phi_0)
                print(f'{label} \t {x}')

    def predict_one(self, phi_0):
        phi = self.forward_nn(phi_0)
        score = phi[len(self.paras)]
        return (1 if score >= 0 else -1)




if __name__ == '__main__':
    input_file = '../data/titles-en-train.labeled'
    to_predict = '../data/titles-en-test.word'
    to_cal_accu = '../data/titles-en-test.labeled'


    nn = NN(1,2,1)
    file = nn.load_df(input_file)
    file.to_csv('/preprossed_data.txt', columns=['label', 'sentence'], sep='\t', header=False, index=False)
    x_y_list = nn.store_features('/preprocessed_data.txt')
    nn.train_nn(1, 0.1)
    nn.predict_all()
