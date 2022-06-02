import numpy as np
from collections import defaultdict

class NeuralNet:
    def __init__(self, layer_num, node_num):
        self.ids = defaultdict(lambda : len(self.ids))#素性を整数IDにして返してくれる
        self.feat_lab = []#素性
        self.net = []#重みとバイアス項でNNを表現
        self.layer_num = layer_num#隠れ層の数
        self.node_num = node_num#隠れ層のノードの数

    def train_nn(self, train_file, iter, lamb):
        #素性辞書の用意
        self.prepare_dict(train_file)
        #NNの初期化
        self.init_net()
        #学習
        for i in range(iter):
            for phi_0, y in self.feat_lab:
                phi = self.forward_nn(phi_0)#素性から各層の値を求める
                delta_d = self.backward_nn(phi, int(y))#逆伝播で各層の誤差を求める
                self.update_weights(phi, delta_d, lamb)
                
    def prepare_dict(self, train_file):
        """入力ファイルから素性の辞書と配列を作成"""
        with open(train_file, "r") as f:
            input_data = [] #正解ラベル，入力文を格納する予定
            for line in f:
                y, x = line.strip().split("\t")#正解ラベル,入力文はタブ区切りになってる
                words = x.split()
                for word in words:
                    self.ids["UNI:"+word]#素性IDの辞書作成（素性の数を数える）
                input_data.append([x, int(y)])
            for x, y in input_data:
                self.feat_lab.append([self.create_features(x), int(y)])#feat_lab[i] = [phi_0, 正解ラベル]

    def create_features(self, x):
        """入力が文，出力が素性"""
        phi = [0]*len(self.ids)
        words = x.split()
        for word in words:
            if "UNI:"+word in self.ids:
                phi[self.ids["UNI:"+word]] += 1 #phi[素性のID] = 素性の出現回数
        return phi

    def init_net(self):
        """入力層，隠れ層，出力層それぞれの重みとバイアスを初期化(-0.5~0.5)"""
        #入力層
        w_in = np.random.rand(self.node_num, len(self.ids)) - 0.5  #素性数×ノード数の配列（入力->隠れ層1の線の数）
        b_in = np.random.rand(self.node_num) - 0.5  #バイアス項はノードごとに足されるからノード数分の配列
        self.net.append([w_in, b_in])

        #隠れ層
        for layer in range(self.layer_num):#層の数だけ
            w_h = np.random.rand(self.node_num, self.node_num) - 0.5  #ノード数×ノード数の配列（隠れ層n->隠れ層n+1の線の数）
            b_h = np.random.rand(self.node_num) - 0.5
            self.net.append([w_h, b_h])

        #出力層
        w_out = np.random.rand(1, self.node_num) - 0.5  #ノード数×出力1つ(1or-1)の配列
        b_out = np.random.rand(1)  #出力は1つだからバイアス項も1つ
        self.net.append([w_out, b_out])

    def forward_nn(self, phi_0):
        """NNの順伝播(入力->出力へ素性の値を計算)"""
        phi = [0]*len(self.ids)
        phi[0] = phi_0#まず入力層の値
        for i in range(len(self.net)):#self.netは層の数に対応
            w, b = self.net[i]#net[その層] = その層の重み,　その層のバイアス
            phi[i+1] = np.tanh(np.dot(w, phi[i]) + b)#前の層の値(phi[i])と重みの内積にバイアス項を足して計算
        return phi

    def backward_nn(self, phi, y):
        """NNの逆伝播(出力->入力へ誤差の勾配を計算)"""
        L = len(self.net)#層の数

        #L+1個の配列
        delta = [0 for _ in range(L)]#誤差
        delta.append(y - phi[L])#y-phi[L]:正解ラベルと出力の誤差
        delta_d = [0 for _ in range(L+1)]

        for i in reversed(range(L)):#逆から
            delta_d[i+1] = delta[i+1] * (1 - phi[i+1] ** 2)#その層の誤差に活性化関数(tanh)の微分をかける
            w, b = self.net[i]#前の層の重みとバイアスを取得
            delta[i] = np.dot(delta_d[i+1],w)#内積使って前の層の誤差を求める
        return delta_d

    def update_weights(self, phi, delta_d, lamb):
        """重みを更新"""
        for i in range(len(self.net)):
            w, b = self.net[i]
            w += lamb * np.outer(delta_d[i+1], phi[i])#次の層のδ'とその層のphiの外積から重みの勾配を計算してw更新
            b += lamb * delta_d[i+1]#バイアスはw関係ないからそのまま学習率λかけるだけ
            self.net[i] = (w, b)
        
    def predict_one(self, phi_0):
        """パーセプトロンの予測"""
        phi = [0]*len(self.ids)
        phi[0] = phi_0
        for i in range(len(self.net)):
            w, b = self.net[i]
            phi[i+1] = np.tanh(np.dot(w, phi[i]) + b)
        score = phi[len(self.net)][0]
        if score >= 0:
            return 1
        else:
            return -1  #最後の出力層の値が0以上なら1，0未満なら-1

    def test_nn(self, test_file, ans_file):
        """予測"""
        with open(test_file, "r") as test_f, open(ans_file,"w") as ans_f:
            for line in test_f:
                x = line.strip()
                phi_0 = self.create_features(x)
                y_pred = self.predict_one(phi_0)
                print(y_pred, file=ans_f)


if __name__ == "__main__":
    path = "../../../nlptutorial/"
    input_file = path + "test/03-train-input.txt"#テスト用
    train_file = path + "data/titles-en-train.labeled"
    test_file = path + "data/titles-en-test.word"

    iter = 1
    lamb = 0.1
    layer_num = 1
    node_num = 2

    for i in range(5):
        ans_file = ans_file = f"my_answer{i+1}.txt"
        nn = NeuralNet(layer_num, node_num)
        nn.train_nn(train_file, iter, lamb)
        nn.test_nn(test_file, ans_file)

"""
Accuracy = 90.010627%
Accuracy = 89.302161%
Accuracy = 90.223167%
Accuracy = 90.967056%
Accuracy = 89.691817%

Average_acc = 90.0389656%
"""
