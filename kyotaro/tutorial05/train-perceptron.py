from collections import defaultdict
import sys

class TrainPerceptron:
    def __init__(self):
        return
    
    def train(self, n, file_name):  # 全体の学習
        w = defaultdict(lambda: 0)
        for l in range(n):
            with open(file_name, "r") as data:
                for line in data:
                    y, x = line.strip().split("\t")
                    phi = self.create_features(x)
                    y_p = self.predict_one(w, phi)
                    if y_p != int(y):
                        self.update_weight(w, phi, y)
        return w

    def update_weight(self, w, phi, y):  # 重みの更新
        for name, value in phi.items():
            w[name] += value * int(y)
    
    def create_features(self, x):  # 単語分割して素性作成
        phi = defaultdict(lambda: 0)
        words = x.split()
        for word in words:
            phi["UNI:" + word] += 1
        return phi
    
    def predict_one(self, w, phi):  # 1文で予測
        score = 0
        for name, value in phi.items():
            if name in w:
                score += value * int(w[name])
        if score >= 0:  # 0を閾値として判別
            return 1
        else:
            return -1

if __name__ == "__main__":
    t = TrainPerceptron()
    file_name = sys.argv[1]
    train = t.train(100, file_name)
    for key, value in sorted(train.items()):
        print(key + "\t" + '{:.6f}'.format(value))