from collections import defaultdict
import numpy as np


class NN:
    def __init__(self):
        self.ids = defaultdict(lambda: len(self.ids))
        self.train_data = []
        self.feat_lab = []
        self.net = []

    def train_input(self, train_file):
        with open(train_file, "r") as t_file:
            for line in t_file:
                y, sentence = line.strip().split("\t")
                self.train_data.append((y, sentence))
                words = sentence.strip().split(" ")
                for word in words:
                    self.ids["UNI:"+word]  # keyを生成

    def create_features(self, sentence):
        phi = [0 for _ in range(len(self.ids))]
        words = sentence.strip().split()
        for word in words:
            phi[self.ids["UNI:"+word]] += 1
        return phi

    def nn_init(self, layer, node):
        self.net = []
        # 入力層
        w_input = np.random.rand(node, len(self.ids)) - 0.5
        b_input = np.random.rand(node)
        self.net.append((w_input, b_input))

        # 中間層
        while len(self.net) < layer:
            w_mid = np.random.rand(node, node) - 0.5
            b_mid = np.random.rand(node)
            self.net.append((w_mid, b_mid))

        # 出力層
        w_output = np.random.rand(1, node) - 0.5
        b_output = np.random.rand(1)
        self.net.append((w_output, b_output))

    def forward_nn(self, phi_0):
        phi = [0 for _ in range(len(self.ids)+1)]
        phi[0] = phi_0
        for i in range(len(self.net)):
            w, b = self.net[i]
            phi[i+1] = np.tanh(np.dot(w, phi[i])+b)
        return phi

    def backward_nn(self, phi, y_pred):
        J = len(self.net)
        delta = [0 for _ in range(J)]
        delta.append(y_pred - phi[J])
        delta_d = [0 for _ in range(J+1)]
        for i in range(J-1, -1, -1):
            delta_d[i+1] = delta[i+1] * (1 - phi[i+1]**2)
            w, b = self.net[i]
            delta[i] = np.dot(delta_d[i+1], w)
        return delta_d

    def update_weights(self, phi, delta_d, lam):
        for i in range(len(self.net)):
            w, b = self.net[i]
            w += lam * np.outer(delta_d[i+1], phi[i])
            b += lam * delta_d[i+1]
            # self.net[i] = w,b

    def save_net(self):
        with open("07net.txt", "w") as f:
            for w, b in self.net:
                f.write(f"{w} {b}\n")

    def save_id(self):
        with open("07id.txt", "w") as f:
            for w, c in self.ids.items():
                f.write(f"{w} {c}\n")

    def nn_learn(self, train_file, iter, layer, node, lam):
        # モデルの読み込み，素性獲得
        self.train_input(train_file)
        for y, sentence in self.train_data:
            self.feat_lab.append((self.create_features(sentence), int(y)))

        # NNの初期化
        self.nn_init(layer, node)

        # 学習
        for _ in range(iter):
            for phi_0, y in self.feat_lab:
                phi = self.forward_nn(phi_0)
                delta_d = self.backward_nn(phi, int(y))
                self.update_weights(phi, delta_d, lam)

        # 重みを保存
        # self.save_net()
        # self.save_id()

    def create_feartures_test(self, sentence):
        phi = [0 for _ in range(len(self.ids))]
        words = sentence.strip().split()
        for word in words:
            if "UNI:" + word in self.ids:
                phi[self.ids["UNI:"+word]] += 1
        return phi

    def predict_one(self, phi_0):
        phi = [0 for _ in range(len(self.net)+1)]
        phi[0] = phi_0
        for i in range(len(self.net)):
            w, b = self.net[i]
            phi[i+1] = np.tanh(np.dot(w, phi[i])+b)
        score = phi[len(self.net)][0]

        if score >= 0:
            return 1
        return -1

    def nn_test(self, test_file, output_file):
        with open(test_file, "r") as t_file, open(output_file, "w")as o_file:
            for line in t_file:
                phi = self.create_feartures_test(line)
                y_pred = self.predict_one(phi)
                o_file.write(f"{y_pred}\n")


if __name__ == "__main__":
    train_file = "test/03-train-input.txt"
    iter = 1
    layer = 1
    node = 2
    lam = 0.1
    # nn_1 = NN()
    # nn_1.nn_learn(train_file, iter, layer, node, lam)

    nn_2 = NN()
    train_file = "data/titles-en-train.labeled"
    test_file = "data/titles-en-test.word"
    output_file = "my_answer"
    nn_2.nn_learn(train_file, iter, layer, node, lam)
    nn_2.nn_test(test_file, output_file)

"""
Accuracy = 91.569253%
"""
