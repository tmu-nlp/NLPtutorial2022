import numpy as np
from tqdm import tqdm
from collections import defaultdict

class RNN:
    def __init__(self, node):
        self.node = node
        self.vocab_ids = defaultdict(lambda : len(self.vocab_ids))#単語(整数ID)
        self.pos_ids = defaultdict(lambda : len(self.pos_ids))#品詞(整数ID)
        self.feat_lab = []
        np.random.seed(7)#シードをあらかじめ0に指定

    def init_rnn(self, input_file):
        """ネットワークをランダムな値で初期化"""
        input_data = []
        with open(input_file, "r") as f:
            for line in f:
                input_data.append(line)
                self.create_ids(line)#素性ID作成
            for line in input_data:
                sentence = self.create_features(line)#素性のベクトル作成
                self.feat_lab.append(sentence)

        #初期値は[-0.1, 0.1]
        #入力の重み
        self.w_rx = np.random.rand(self.node, len(self.vocab_ids))/5 - 0.1
        #隠れ状態の重み
        self.w_rh = np.random.rand(self.node, self.node)/5 - 0.1
        self.b_r = np.random.rand(self.node)/5 - 0.1
        #出力の重み
        self.w_oh = np.random.rand(len(self.pos_ids), self.node)/5 - 0.1
        self.b_o = np.random.rand(len(self.pos_ids))/5 - 0.1

    def create_ids(self, x):
        """入力文xから素性(ID)を作成"""
        words = x.strip().split()
        for word in words:
            x, y = word.split("_")#単語_品詞になってる
            self.vocab_ids[x]
            self.pos_ids[y]

    def create_features(self, x):
        """素性IDから素性のone-hotベクトルを取得し，文ごとで出力"""
        sentence = []
        words = x.strip().split()
        for word in words:
            x, y = word.split("_")
            x_vec = self.create_one_hot(self.vocab_ids[x], len(self.vocab_ids))
            y_vec = self.create_one_hot(self.pos_ids[y], len(self.pos_ids))
            sentence.append([x_vec, y_vec])#単語&品詞のベクトルをペアで保存
        return sentence

    def create_one_hot(self, id, size):
        """one-hotベクトルの作成"""
        vec = np.zeros(size)
        vec[id] = 1
        return vec

    def find_max(self, p):
        """確率分布pから確率最大のインデックスyを探す"""
        return np.argmax(p)

    def soft_max(self, x):
        """0から1の間の値に変換(多クラス)"""
        wa = np.sum(np.exp(x))
        p = np.exp(x)/wa
        return p
    
    def forward_rnn(self, sentence):
        """RNNの前向き計算"""
        h = []#各時間tにおける隠れ層の値
        p = []#各時間tにおける出力の確率分布の値
        y = []#各時間tにおける出力の確率分布の値??
        for t in range(len(sentence)):
            x, _ = sentence[t]
            if t > 0:
                h.append(np.tanh(np.dot(self.w_rx, x) + np.dot(self.w_rh, h[t-1]) + self.b_r))
            else:#時刻０の時はt-1の状態を使えない
                h.append(np.tanh(np.dot(self.w_rx, x) + self.b_r))
            p.append(self.soft_max(np.dot(self.w_oh, h[t]) + self.b_o))
            y.append(self.find_max(p[t]))
        return h, p, y

    def gradient_rnn(self, sentence, p, h):#あんまわかってない
        """RNNノ完全勾配計算"""
        #0で初期化された重みとバイアスを各層分用意
        w_rx_diff = np.zeros((self.node, len(self.vocab_ids)))
        w_rh_diff = np.zeros((self.node, self.node))
        b_r_diff = np.zeros(self.node)
        w_oh_diff = np.zeros((len(self.pos_ids), self.node))
        b_o_diff = np.zeros(len(self.pos_ids))
        delta_r_d = np.zeros(self.node)#次の時間から伝播するエラー
        for t in range(len(sentence))[::-1]:
            x, y_d = sentence[t]#x:単語のベクトル，y_d:品詞のベクトル
            #出力層エラー
            delta_o_d = y_d - p[t]
            #出力層重み勾配
            w_oh_diff += np.outer(delta_o_d, h[t])
            b_o_diff += delta_o_d
            #逆伝播
            delta_r = np.dot(delta_r_d, self.w_rh) + np.dot(delta_o_d, self.w_oh)
            #tanhの勾配
            delta_r_d = delta_r*(1 - h[t]**2)
            #隠れ層の重み勾配
            w_rx_diff += np.outer(delta_r_d, x)
            b_r_diff += delta_r_d
            if t != 0:
                w_rh_diff += np.outer(delta_r_d, h[t-1])
        return [w_rx_diff, w_rh_diff, b_r_diff, w_oh_diff, b_o_diff]

    def update_weights(self, delta, lamb):
        """求めた勾配(delta)で重み更新"""
        w_rx_diff, w_rh_diff, b_r_diff, w_oh_diff, b_o_diff = delta
        
        self.w_rx += lamb * w_rx_diff
        self.w_rh += lamb * w_rh_diff
        self.b_r += lamb * b_r_diff
        self.w_oh += lamb * w_oh_diff
        self.b_o += lamb * b_o_diff
    
    def train_rnn(self, input_file, lamb, iter):
        """RNNを学習"""
        #ネットワークを初期化
        self.init_rnn(input_file)
        #学習を行う
        for i in tqdm(range(iter)):
            for sentence in self.feat_lab:
                h, p, y = self.forward_rnn(sentence)
                delta = self.gradient_rnn(sentence, p, h)
                self.update_weights(delta, lamb)

    def create_features_test(self, x):
        """評価データ用の素性作成関数"""
        phi = []
        words = x.strip().split()#入力文を単語に分ける
        for word in words:
            if word in self.vocab_ids.keys():#既知語の場合はその単語のone-hotベクトルを取得
                phi.append(self.create_one_hot(self.vocab_ids[word], len(self.vocab_ids)))
            else:#未知語の場合は語彙サイズ長のベクトル(値は全て0)を取得
                phi.append(np.zeros(len(self.vocab_ids)))
        return phi

    def forward_rnn_test(self, sentence):
        """評価用のRNNの前向き計算"""
        h = []#各時間tにおける隠れ層の値
        p = []#各時間tにおける出力の確率分布の値
        y = []#各時間tにおける出力の確率分布の値??
        for t in range(len(sentence)):
            x = sentence[t]
            if t > 0:
                h.append(np.tanh(np.dot(self.w_rx, x) + np.dot(self.w_rh, h[t-1]) + self.b_r))
            else:#時刻０の時はt-1の状態を使えない
                h.append(np.tanh(np.dot(self.w_rx, x) + self.b_r))
            p.append(self.soft_max(np.dot(self.w_oh, h[t]) + self.b_o))
            y.append(self.find_max(p[t]))
        return h, p, y


    def test_rnn(self, test_file, ans_file):
        with open(test_file, "r") as test_f, open(ans_file, "w") as ans_f:
            for line in test_f:
                sentence = []
                line = line.strip()
                phi = self.create_features_test(line)
                h, p, y_pred = self.forward_rnn_test(phi)
                for ans in y_pred:
                    for pos, id in self.pos_ids.items():
                        if id == ans:
                            sentence.append(pos)
                            break
                print(" ".join(sentence), file=ans_f)


if __name__ == "__main__":
    path = "../../../nlptutorial/"
    #テスト用
    #train_file = path + "test/05-train-input.txt"
    #test_file = path + "test/05-test-input.txt"

    train_file = path + "data/wiki-en-train.norm_pos"
    test_file = path + "data/wiki-en-test.norm"

    #パラメータ大事っぽい？？
    node = 20#もっと増やした方がいい？
    lamb = 0.01#これは最適ぽい？
    iter = 7#これより上だと正解率が大幅に下がる，なぜ？？
    rnn = RNN(node)
    rnn.train_rnn(train_file, lamb, iter)
    ans_file = "my_answer.txt"
    rnn.test_rnn(test_file, ans_file)

"""
Accuracy: 87.00% (3970/4563)

Most common mistakes:
NN --> NNP      52
JJ --> NN       43
JJ --> NNP      32
NNS --> NN      32
VBN --> JJ      26
NNS --> NNP     23
NN --> NNS      21
NNP --> NN      20
VBN --> NN      12
RB --> NNP      12
"""