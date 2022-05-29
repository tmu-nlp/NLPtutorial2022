from collections import defaultdict
import sys
import numpy as np
import math

class SVM:
    def __init__(self):
        self.margin = 5  # 許容範囲
        self.c = 1.0  # 正則化係数
        self.last = defaultdict(lambda: 0)  # keyを最後に変更した時の番号がvalueになる
        self.times = 1
        self.iter = 0
    
    def predict_all(self, file_name):
        """
        ラベルの予測
        """
        w = self.online_learning(file_name)  # 重みベクトルを学習
        with open("titles-en-test.word", "r") as test:
            for x in test:
                x = x.strip()
                phi = self.create_features(x)  # 素性抽出
                y_p = self.predict_one(w, phi)  # sign(w * phi)を計算
                print(str(y_p) + "\t" + x)  # 予測したラベルと原文表示

    def online_learning(self, file_name):
        """
        学習
        """
        w = defaultdict(lambda: 0)  # 重みベクトルの初期化
        for l in range(self.times):  # 学習回数設定
            with open(file_name, "r") as data:
                for line in data:
                    y, x = line.strip().split("\t")  # yに正解ラベル、xに原文を代入
                    phi = self.create_features(x)  # 素性抽出
                    val = self.calculate_value(w, phi, y)  # スコアを計算
                    if val <= self.margin:  # marginより小さい場合
                        #self.update_weight(w, phi, y)  # 重み更新
                        self.update_weight_efficientry(phi, w, y)
                        #self.update_weight_L2(w, phi, y)
        return w
    
    def create_features(self, x):
        """
        素性抽出
        """
        phi = defaultdict(lambda: 0)
        words = x.split(" ")  # 単語に分解
        for word in words:
            phi["UNI:" + word] += 1  # 単語の頻度を辞書に追加
        return phi
    
    def update_weight(self, w, phi, y):
        """
        重み更新（効率は良くない）
        """
        for name, value in w.items():
            if abs(value) <= self.c:  # 正則化
                w[name] = 0
            else:
                w[name] -= self.sign(value) * self.c  # 値が0より大きければ正則化係数を引き、値が0より小さければ正則化係数を足す
        for name, value in phi.items():
            w[name] += value * int(y)  # 正解方向に重みをずらす
    
    def sign(self, value):
        """
        符号関数
        """
        if value >= 0:
            return 1
        else:
            return -1
    
    def calculate_value(self, w, phi, y):
        """
        境界線との距離を計算
        """
        score = 0
        for name, value in phi.items():
            if name in w:
                score += int(w[name]) * value  # 正しかったらscoreにたす
        score *= int(y)
        return score
    
    def predict_one(self, w, phi):  # 1文で予測
        score = 0
        for name, value in phi.items():
            if name in w:
                score += value * int(w[name])
        if score >= 0:  # 0を閾値として判別
            return 1
        else:
            return -1

    def getw(self, w, name):
        """
        効率の良い正則化
        """
        if self.iter != self.last[name]:
            c_size = self.c * (self.iter - self.last[name])  # １つずつではなく、最後に変更したところからまとめて変更する
            if abs(w[name]) <= c_size:
                w[name] = 0
            else:
                w[name] -= self.sign(w[name]) * c_size
            self.last[name] = self.iter
        return w[name]
    
    def update_weight_efficientry(self, phi, w, y):
        """
        効率の良い正則化を用いた学習
        """
        for name, value in phi.items():
            w[name] = self.getw(w, name)
            w[name] += value * int(y)
        self.iter += 1
    
    
    # def update_weight_L2(self, w, phi, y):
    #     """
    #     L2正則化を行った重み更新
    #     """
    #     a = 0
    #     for name, value in w.items():
    #         a += value*value
    #     l2_norm = math.sqrt(a)
    #     for name, value in w.items():
    #         w[name] = w[name] / l2_norm
    #     for name, value in phi.items():
    #         w[name] += value * int(y)
    

if __name__ == "__main__":
    file_name = sys.argv[1]
    s = SVM()
    s.predict_all(file_name)