#!/opt/homebrew/bin/python3
from collections import defaultdict
import sys
from nltk.corpus import stopwords
import nltk

"""
第一引数:訓練か予測かを選択
train -> 訓練
test -> 予測

第二引数:訓練回数
int : n ->　n回繰り返したモデルを訓練/使用 
"""

# https://nashidos.hatenablog.com/entry/2020/08/12/205119
nltk.download('stopwords')
stop_words = stopwords.words('english')


class Perceptron():
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
            if word not in stop_words:
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

    def update_weight(self, w, phi, y):
        for name, value in phi.items():
            w[name] += value * int(y)

    def online_learning(self, train_file, l):
        w = defaultdict(lambda: 0)
        for i in range(l):
            with open(train_file, 'r') as t_file:
                for line in t_file:
                    y, x = line.strip().split("\t")
                    phi = self.create_features(x)
                    y_pred = self.predict_one(w, phi)
                    if int(y_pred) != int(y):
                        self.update_weight(w, phi, y)
        return w

    def weight_output(self, w, output_file):
        with open(output_file, 'w') as o_file:
            for key, value in w.items():
                o_file.write(f"{key} {value}\n")

    def load_file(self, model_file):
        w = defaultdict(lambda: 0)
        with open(model_file, 'r')as m_file:
            for line in m_file:
                key, value = line.split(" ")
                w[key] = float(value)
        return w


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train_num = sys.argv[2]
        model_file = "data/titles-en-train.labeled"
        output_file = "model-05-prep-" + train_num + ".txt"
        train = Perceptron()
        weight = train.online_learning(model_file, int(train_num))
        train.weight_output(weight, output_file)

    elif sys.argv[1] == "test":
        train_num = sys.argv[2]
        model_file = "model-05-prep-" + train_num + ".txt"
        input_file = "data/titles-en-test.word"
        output_file = "my_answer05-prep-" + train_num
        test = Perceptron()
        test.predict_all(model_file, input_file, output_file)
