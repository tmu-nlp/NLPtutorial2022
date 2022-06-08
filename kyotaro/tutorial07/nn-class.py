from collections import defaultdict
import numpy as np


class NeuralNet:
    def __init__(self):
        """変数の初期化"""
        self.ids = defaultdict(lambda: len(self.ids))  # 各特徴量のIDが入っている辞書
        self.feat_lab = []  # 特徴量の状態と正解ラベルが入っている
        self.hidden_num = 2  # 隠れ層のノードの数
        self.net = []  # 重みとバイアスが入っているリスト
        self.lmd = 0.1  # 学習率
        self.layer = 1  # 隠れ層の数


###ここより下でパラメータの設定###


    def initialize(self, file_name):
        """IDを作って正解ラベルとくっつける"""
        with open(file_name, "r") as data:
            sen_lab = []
            for line in data:
                y, x = line.strip().split("\t")  # yに正解ラベル、xに文が入っている
                self.make_id(x)  # 各特徴量にIDを振り分け
                sen_lab.append((x, int(y)))
            self.initialize_net()  # パラメータの初期化
            for sen, lab in sen_lab:
                self.add_feat(sen, lab)

    def make_id(self, x):
        """ID振り分け"""
        words = x.split()
        for word in words:
            self.ids["UNI:" + word]

    def add_feat(self, x, y):
        """素性と正解ラベルが入っているリストを生成"""
        self.feat_lab.append((self.create_fetures(x), int(y)))

    def create_fetures(self, x):
        """素性を作る"""
        phi = [0 for _ in range(len(self.ids))]
        words = x.split()
        for word in words:
            if f'UNI:{word}' in self.ids:
                phi[self.ids["UNI:"+word]] += 1
        return phi

    def initialize_net(self):
        """重みとバイアスの初期化（理解のために次元が１の場合も記載）"""

        # 入力層から最初の隠れ層までの重みとバイアス
        w_0 = np.random.rand(self.hidden_num, len(self.ids)) / 5 - 0.1
        b_0 = np.random.rand(self.hidden_num) / 5 - 0.1
        self.net.append((w_0, b_0))

        # 最初の隠れ層から出力層の前の層までの重みとバイアス
        for _ in range(self.layer - 2):
            w_h = np.random.rand(self.hidden_num, self.hidden_num) / 5 - 0.1
            b_h = np.random.rand(self.hidden_num) / 5 - 0.1
            self.net.append((w_h, b_h))

        # 出力層での重みとバイアス
        w_N = np.random.rand(1, self.hidden_num) / 5 - 0.1
        b_N = np.random.rand(1) / 5 - 0.1
        self.net.append((w_N, b_N))


###ここより下の関数を用いて学習する###


    def online_learning(self, n, file_name):
        """学習"""
        self.initialize(file_name)
        for l in range(n):
            for phi_0, y_t in self.feat_lab:
                phis = self.forward_nn(phi_0)
                delta_dash = self.backward_nn(phis, y_t)
                self.update_weight(phis, delta_dash)

    def forward_nn(self, phi_0):
        """入力層から出力層までの道のり"""
        phis = [0 for _ in range(len(self.ids) + 1)]
        phis[0] = phi_0  # 各層の値をphi_0に設定
        for i in range(len(self.net)):  # １層づつ進めていく
            w, b = self.net[i]  # 重みとバイアスを抽出
            # 内積をとって非線形関数をかける（次の隠れ層に移動）
            phis[i + 1] = np.tanh(np.dot(w, phis[i]) + b)
        return phis

    def backward_nn(self, phis, y_t):
        """重みの更新のために逆順で見た時の勾配が必要"""
        J = len(self.net)  # 層から層へ変換する数
        delta = [0 for _ in range(J + 1)]  # 勾配を０に初期化
        delta[J] = y_t - phis[J][0]  # 一番後ろの勾配を設定、内積を取るのでnp.array()
        delta_dash = [0 for _ in range(J + 1)]  # 一個前の勾配
        for i in reversed(range(J)):  # 後ろから見ていく
            delta_dash[i + 1] = delta[i + 1] * \
                (1 - phis[i + 1] ** 2)  # 非線形関数を取り除く
            w, b = self.net[i]  # 重みとバイアスの更新
            delta[i] = np.dot(delta_dash[i+1], w)  # 勾配をずらす
        return delta_dash

    def update_weight(self, phis, delta_dash):
        """重みの更新"""
        for i in range(len(self.net)):
            w, b = self.net[i]
            w += self.lmd * np.outer(delta_dash[i + 1], phis[i])
            b += self.lmd * delta_dash[i + 1]


###ここより下でテスト###

    def predict_all(self):
        """ファイル全体の予測"""
        with open("titles-en-test.word", "r") as data:
            for line in data:
                x = line.strip()
                phi0 = self.create_fetures(x)  # 素性抽出
                label = self.predict_one(phi0)  # 一文の予測
                print(f'{label}\t{x}')

    def predict_one(self, phi0):
        """一文の予測"""
        phis = self.forward_nn(phi0)  # 素性を前に進める（違う点にマッピング）
        score = phis[len(self.net)]  # 一番後ろの素性を最後のスコアにする
        return (1 if score >= 0 else -1)


if __name__ == "__main__":
    file_name = "titles-en-train.labeled"
    np.random.seed(seed=1013)
    nn = NeuralNet()
    nn.online_learning(1, file_name)
    nn.predict_all()


"""
n = 1, c = 0.01
Accuracy = 92.490259%

n = 1, c = 0.1
Accuracy = 92.383989%
"""
