#!/opt/homebrew/bin/python3
from collections import defaultdict
import sys
import numpy as np

"""
第一引数:訓練か予測かを選択
train -> 訓練
test -> 予測

第二引数:訓練回数
int : n ->　n回繰り返したモデルを訓練/使用 
"""


class Perceptron():
    def __init__(self):
        self.c = 1.0
        self.last = defaultdict(lambda: 0)
        self.iter = 0

    def predict_one(self, w, phi):
        score = 0
        for name, value in phi.items():
            if name in w:
                score += value * w[name]
        if score >= 0:
            return 1
        else:
            return -1

    def create_features(self, x):
        phi = defaultdict(lambda: 0)
        words = x.split(" ")
        for word in words:
            phi[f"UNI:{word}"] += 1
        return phi

    def predict_all(self, model_file, input_file, output_file):
        w = self.load_file(model_file)
        with open(output_file, 'w') as o_file:
            with open(input_file, 'r') as i_file:
                for x in i_file:
                    x = x.strip()
                    phi = self.create_features(x)
                    y_pred = self.predict_one(w, phi)
                    o_file.write(f"{y_pred}\n")

    def weight_output(self, w, output_file):
        with open(output_file, 'w') as o_file:
            for key, value in w.items():
                self.getw(w, key)
                o_file.write(f"{key} {value}\n")

    def load_file(self, model_file):
        w = defaultdict(lambda: 0)
        with open(model_file, 'r')as m_file:
            for line in m_file:
                key, value = line.split(" ")
                w[key] = float(value)
        return w

    def predict_one_margin(self, w, phi, y):
        score = 0
        for name, value in phi.items():
            if name in w:
                score += value * w[name] * int(y)
        return score

    def update_weight_l1(self, w, phi, y):
        for name, value in w.items():
            if abs(value) < self.c:
                w[name] = 0
            else:
                w[name] -= np.sign(int(value)) * self.c
        for name, value in phi.items():
            w[name] += value * int(y)

    def online_learning_margin(self, train_file, l, margin):
        w = defaultdict(lambda: 0)
        for i in range(l):
            with open(train_file, 'r') as t_file:
                for line in t_file:
                    y, x = line.strip().split("\t")
                    phi = self.create_features(x)
                    val = self.predict_one_margin(w, phi, y)
                    if val <= margin:
                        self.update_weight_l1(w, phi, y)
        return w

    def get_val(self, w, phi, y):
        score = 0
        for name, value in phi.items():
            score += value * w[name]
        score *= int(y)
        return score

    def update_weight_l1_opt(self, w, phi, y):
        for name, value in phi.items():
            w[name] = self.getw(w, name)
            w[name] += value * int(y)
        self.iter += 1

    def online_learning_margin_opt(self, train_file, l, margin):
        w = defaultdict(lambda: 0)
        for i in range(l):
            with open(train_file, 'r') as t_file:
                for line in t_file:
                    y, x = line.strip().split("\t")
                    phi = self.create_features(x)
                    val = self.get_val(w, phi, y)
                    if val <= margin:
                        self.update_weight_l1_opt(w, phi, y)
        return w

    def getw(self, w, name):
        if self.iter != self.last[name]:
            c_size = self.c * (self.iter - self.last[name])
            if abs(w[name]) <= c_size:
                w[name] = 0
            else:
                w[name] -= np.sign(w[name]) * c_size
            self.last[name] = self.iter
        return w[name]

    def set_c(self, c):
        self.c = c

    def get_c(self):
        return self.c

    def last_output(self):
        for key, value in self.last.items():
            print(f"{key} : {value}")


if __name__ == "__main__":
    margin = 0
    if sys.argv[1] == "train":
        train_num = sys.argv[2]
        model_file = "data/titles-en-train.labeled"
        output_file = "model-06-" + train_num + "-" + str(margin) + ".txt"
        train = Perceptron()
        weight = train.online_learning_margin_opt(
            model_file, int(train_num), margin)
        train.weight_output(weight, output_file)
        # train.last_output()
        print(f"margin = {margin}")
        print(f"c = {train.get_c()}")
        print(f"train_num = {train_num}")

    elif sys.argv[1] == "test":
        train_num = sys.argv[2]
        model_file = "model-06-" + train_num + "-" + str(margin) + ".txt"
        input_file = "data/titles-en-test.word"
        output_file = "my_answer06-" + train_num + "-" + str(margin)
        test = Perceptron()
        test.predict_all(model_file, input_file, output_file)
