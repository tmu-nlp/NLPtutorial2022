#実行コード
#python tutorial07.py ../../../nlptutorial/data/titles-en-train.labeled ids.txt weight.txt ../../../nlptutorial/data/titles-en-test.word result.txt
#python2 ../../../nlptutorial/script/grade-prediction.py ../../../nlptutorial/data/titles-en-test.labeled result.txt
import numpy as np
import argparse
from collections import defaultdict
import time

np.random.seed(0) #再現性

class NeuralNet:
    def __init__(self):
        self.ids = defaultdict(lambda: len(self.ids))  # len(ids)がiでids["hoge"]をしたとき、hogeがidsにないならids["hoge"]はiになり、len(ids)はi+1になる
        self.network = [] #重み。self.network[i][0]がi層目のwで、self.network[i][1]がi層目のb
        self.labmda = 0.01 #学習率
        self.hidden_layer = 1 #隠れ層の数
        self.hidden_node = 2 #隠れ層のノードの数


    def train_nn(self, train_file, times):
        with open(train_file, "r") as f_train:
            data_train = f_train.readlines()

        #訓練データ内に出てくる単語にIDを割り当てる。各行でのphiの長さはlen(self.network[0][0])と同じじゃなきゃダメだからphiを作り始める前にやる
        for line in data_train:
            _, x = line.split('\t')
            for word in x.strip().split():
                self.ids[f'UNI:{word}'] #wordのID作成

        self.initialize_network()

        for _ in range(times):
            for line in data_train:
                y_ans, x = line.strip().split('\t')
                y_ans = int(y_ans)
                phi_0 = self.create_features(x.strip()) #ニューラルネットへの入力となる配列。phi_0[i]はIDがiの単語に対応
                #print(phi_0)
                phi = self.forward_nn(phi_0)
                delta_gradient = self.backward_nn(phi, y_ans)
                self.update_weights(phi, delta_gradient)


    def initialize_network(self):
        #重み作成（重みの初期値を-0.1~0.1にする）
        w1 = np.random.rand(self.hidden_node, len(self.ids)) / 5 - 0.1
        b1 = np.random.rand(self.hidden_node)
        self.network.append([w1, b1])

        for _ in range(self.hidden_layer - 1): #隠れ層が2以上のとき
            wi = np.random.rand(self.hidden_node, self.hidden_node) / 5 - 0.1
            bi = np.random.rand(self.hidden_node)
            self.network.append([wi, bi])

        w_last = np.random.rand(1, self.hidden_node) / 5 - 0.1
        b_last = np.random.rand(1) / 5 - 0.1
        self.network.append([w_last, b_last])
        #print(f'network ini\n{self.network}')


    def create_features(self,  x):
        phi = np.zeros(len(self.ids))
        words = x.split()
        for word in words:
            if f'UNI:{word}' in self.ids.keys(): #テスト時、idsの中にその単語がなかったら特徴量として使わない(入力ノードの数が変わってしまうので)
                phi[self.ids[f'UNI:{word}']] += 1
        return phi


    def forward_nn(self, phi_0):
        phi = [0] * (self.hidden_layer + 2) #入力層＋隠れ層の数＋出力層 = 隠れ層の数+2
        phi[0] = phi_0
        #phi = [phi_0]
        for layer in range(len(self.network)):
            w, b = self.network[layer]
            #print(f'w is {w}')
            # print(w, w.shape)
            # print(phi[layer])
            phi[layer + 1] = np.tanh(np.dot(w, phi[layer]) + b) #ひとつ前の出力と重みで計算。
            
        #print(f'phi is {phi}')
        return phi


    def backward_nn(self, phi, y_ans):
        J = len(self.network)
        delta = [0] * (J+1)
        delta[J] = np.array([y_ans - phi[J][0]]) #phi[J]は出力層。array型だから[0]で値を取り出す。つまりこれは正解-出力
        #print(f'delta is {delta}')
        delta_gradient = [0] * (J+1)
        for i in range(J)[::-1]:
            delta_gradient[i+1] = delta[i+1] * (1 - phi[i+1] * phi[i+1])
            #print(f'delta_gradient is {delta_gradient}')
            w, b = self.network[i]
            # print(f'w is {w}')
            # print(type(w), type(delta_gradient[i+1]))
            delta[i] = np.dot(delta_gradient[i+1], w)
        return delta_gradient


    def update_weights(self, phi, delta_gradient):
        for i in range(len(self.network)):
            w, b = self.network[i]
            w += self.labmda * np.outer(delta_gradient[i+1], phi[i])
            b += self.labmda * delta_gradient[i+1]
            self.network[i] = [w, b]


    def predict_all(self, test_file, result_file):
        with open(test_file, "r") as f_test:
            data_test = f_test.readlines()
        with open(result_file, "w") as f_result:
            for line in data_test:
                line = line.strip()
                phi = self.create_features(line)
                y_pred = self.predict_one(phi)
                f_result.write(f'{y_pred}\t{line}\n')


    def predict_one(self, phi):
        phis = self.forward_nn(phi)
        score = phis[len(self.network)][0] #phis[len(self.network)]はニューラルネットの出力層の値。arrayだから[0]で値を取り出す
        if score >= 0:
            return 1
        else:
            return -1


    def output_weights(self, output_file):
        with open(output_file, "w") as f_out:
            for i, (w, b) in enumerate(self.network):
                f_out.write(f'----------{i}層目----------\n{w}\n{b}\n')


    def output_ids(self, output_file_ids):
        with open(output_file_ids, "w") as f_out_ids:
            for word, id in self.ids.items():
                f_out_ids.write(f'{word}\t{id}\n')


if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('train_file', help='訓練データファイル')
    parser.add_argument('ID_output_file', help='訓練データ内で出現する単語に割り当てたIDを出力するファイル')
    parser.add_argument('weight_output_file', help='学習した重みを出力するファイル')
    parser.add_argument('test_file', help='テストデータファイル')
    parser.add_argument('result_file', help='分類結果を出力するファイル')
    args = parser.parse_args()

    iter = 1
    k = NeuralNet()
    k.train_nn(args.train_file, iter)
    k.output_ids(args.ID_output_file)
    k.output_weights(args.weight_output_file)
    k.predict_all(args.test_file, args.result_file)

    end = time.time()
    print(f'iter：{iter}  実行時間：{end - start}[s]')

    """
    隠れ層数=1　隠れnode数=2　学習率0.1
    iter：1  実行時間：5.384057998657227[s]　Accuracy = 92.065179%
    iter：2  実行時間：9.198371171951294[s]　Accuracy = 93.163301%
    iter：5  実行時間：24.0101420879364[s]　Accuracy = 93.907191%

    隠れ層数=1　隠れnode数=2　学習率0.01
    iter：1  実行時間：4.893983840942383[s]　Accuracy = 92.561105%
    iter：2  実行時間：10.662533044815063[s]　Accuracy = 93.411265%
    iter：5  実行時間：24.636680841445923[s]　Accuracy = 93.942614%


    隠れ層数=1　隠れnode数=10　学習率0.01
    iter：5  実行時間：50.562594175338745[s]　Accuracy = 93.942614%

    隠れ層数=10　隠れnode数=2　学習率0.01
    iter：5  実行時間：38.08616924285889[s]　Accuracy = 52.320227%

    隠れ層数=10　隠れnode数=10　学習率0.01
    iter：5  実行時間：66.00151991844177[s]　Accuracy = 52.320227%
    """
