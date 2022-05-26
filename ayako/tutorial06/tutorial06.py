from collections import defaultdict
import numpy as np
import time, sys

class SVM:
    def __init__(self, iteration, margin, c, last):
        self.iteration = iteration
        self.margin = margin
        self.c = c#正則化係数
        self.last = last#最後に重み更新したループを保持する辞書

    #素性作成
    def create_features(self, x):
        phi = defaultdict(lambda : 0)#素性ベクトル，単語の出現回数辞書
        words = x.split()
        for word in words:
            phi["UNI:"+word] += 1
        return phi

    #一つの事例に対する予測
    def predict_one(self, w, phi):
        score = 0
        for name, value in phi.items():
            if name in w:#既知語だったら重み*その単語の頻度の値をスコアに
                score += value * w[name]
        if score >= 0:
            return 1
        else:
            return -1

    #L1正則化を使った重みの更新
    def update_weights(self, w, phi, y):#y=1or-1
        for name, value in w.items():#重み辞書
            if abs(value) < self.c:#重みの絶対値が定数cより小さい場合
                w[name] = 0 #重みを0に
            else:
                #sign関数(符号関数):正の値に1,負の値に-1,0に0を返す
                w[name] -= np.sign(value)*self.c#重みが正ならばc引く，負ならばc足す
        for name, value in phi.items():#既知語の重みを更新
            w[name] += value * y
    
    def update_weights_lazy(self, w, phi, y):#y=1or-1
        for name, value in w.items():#重み辞書
            if abs(value) < self.c:#重みの絶対値が定数cより小さい場合
                w[name] = 0 #重みを0に
            else:
                w[name] = self.getw(w, name)#遅延評価
                w[name] -= np.sign(value)*self.c
        for name, value in phi.items():#既知語の重みを更新
            w[name] += value * y

    #マージンを使うSVM（普通ver）
    def train_svm(self, train_file, model_file):
        w = defaultdict(lambda : 0)#重みの辞書
        with open(model_file, "w") as m_file:
            for i in range(self.iteration):
                with open(train_file, "r") as f:
                    for line in f:
                        y, x = line.strip().split("\t")#正解値と文を取得
                        y = int(y)
                        phi = self.create_features(x)#文に含まれる各単語の辞書を取得
                        #margin(val)を計算
                        val = 0
                        for name, value in phi.items():
                            if name in w:#その素性の重みが既にある場合->ない場合はどうせw=0よりval=0
                                val += value * w[name] * y #val = phi * w * y
                        if val <= self.margin:#マージンよりvalが小さい(確信度が低い)場合，重みを更新
                            self.update_weights(w, phi, y)
            for name, value in sorted(w.items()):#テキストファイルとして中身みたいからdillやめる
                print(f"{name}\t{value}", file=m_file)

    #マージンを使うSVM（遅延評価ver）
    def train_svm_lazy(self, train_file, model_file):
        w = defaultdict(lambda : 0)#重みの辞書
        with open(train_file, "r") as f, open(model_file, "w") as m_file:
            for line in f:
                y, x = line.strip().split("\t")#正解値と文を取得
                y = int(y)
                phi = self.create_features(x)#文に含まれる各単語の辞書を取得
                #margin(val)を計算
                val = 0
                for name, value in phi.items():
                    if name in w:#その素性の重みが既にある場合->ない場合はどうせw=0よりval=0
                        val += value * w[name] * y #val = phi * w * y
                if val <= self.margin:#マージンよりvalが小さい(確信度が低い)場合，重みを更新
                    self.iteration += 1#繰り返し回数をカウント
                    self.update_weights_lazy(w, phi, y)
            for name, value in w.items():
                value = self.getw(w, name)#遅延評価
            for name, value in sorted(w.items()):#テキストファイルとして中身みたいからdillやめる
                print(f"{name}\t{value}", file=m_file)

    #全事例に対する予測
    def predict_all(self, weights, test_file, output_file):
        w = defaultdict(lambda : 0)
        with open(model_file, "r") as m_file:
            for line in m_file:
                name, value = line.strip().split("\t")
                value = float(value)#いる？
                w[name] = value
        with open(test_file, "r") as f, open(output_file, "w") as o_file:
            for x in f.readlines():
                x = x.strip()
                phi = self.create_features(x)
                y_predict = self.predict_one(w, phi)
                print(str(y_predict) + "\t" + x, file=o_file)
    
    def test_svm(self, weights, test_file, output_file):
        self.predict_all(weights, test_file, output_file)
    
    #効率化(遅延評価：値が必要になった時にだけ評価する)
    def getw(self, w, name):
        if self.iteration != self.last[name]:
            c_size = self.c * (self.iteration - self.last[name])
            if abs(w[name]) <= c_size:
                w[name] = 0
            else:
                w[name] -= np.sign(w[name]) * c_size#重みを更新
            self.last[name] = self.iteration#lastを更新
        return w[name]


if __name__ == "__main__":
    path = "../../../nlptutorial/"
    train_file = path + "data/titles-en-train.labeled"
    test_file = path + "data/titles-en-test.word"
    model_file = "weights_dic.txt"
    output_file = "my_answer"
    last = defaultdict(lambda : 0)

    #遅延評価で効率化する場合
    if sys.argv[1] == "lazy":
        svm = SVM(0, 20, 0.0001, last)
        t1 = time.time()
        svm.train_svm_lazy(train_file, model_file)
        t2 = time.time()
        print(f"学習時間（遅延評価ver）:{t2-t1}")
    #効率化なしの場合はコマンドライン引数でイテレータ指定
    else:
        iteration = int(sys.argv[1])
        svm = SVM(iteration, 10, 0.0001, last)
        t1 = time.time()
        svm.train_svm(train_file, model_file)
        t2 = time.time()
        print(f"学習時間（iter={iteration}）:{t2-t1}")

    svm.test_svm(model_file, test_file, f"{output_file}_{sys.argv[1]}.txt")

"""
SVMの学習時間(margin=10)
学習時間（遅延評価ver）:41.21848511695862   Accuracy = 92.136026%
学習時間（iter=1）:14.646186113357544   Accuracy = 92.206872%
学習時間（iter=2）:33.26134395599365    Accuracy = 92.879915%
学習時間（iter=4）:65.31727194786072    Accuracy = 91.958909%   <-????
学習時間（iter=8）:111.61298418045044   Accuracy = 93.269571%   <-????

SVMの学習時間(margin=20)
学習時間（遅延評価ver）:40.28055787086487   Accuracy = 92.136026%
学習時間（iter=1）:18.748382806777954   Accuracy = 93.092455%
学習時間（iter=2）:41.31378388404846    Accuracy = 93.659228%
学習時間（iter=4）:81.7234537601471     Accuracy = 91.640099%   <-????
学習時間（iter=8）:147.03668904304504   Accuracy = 93.234148%   <-????

パーセプトロンと比較
iter    Perceptron              SVM(margin=10)          SVM(margin=20)
1       Accuracy = 90.967056%   Accuracy = 92.206872%   Accuracy = 93.092455%
2       Accuracy = 91.781792%   Accuracy = 92.879915%   Accuracy = 93.659228%
4       Accuracy = 90.825363%   Accuracy = 91.958909%   Accuracy = 91.640099%
8       Accuracy = 91.569253%   Accuracy = 93.269571%   Accuracy = 93.234148%
"""