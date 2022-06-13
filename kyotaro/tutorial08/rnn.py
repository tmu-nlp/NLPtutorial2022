import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score

class RNN():
    def __init__(self):
        self.hidden_num = 8  # 隠れ層のループ
        self.x_ids = defaultdict(lambda: len(self.x_ids))  # 単語のID
        self.y_ids = defaultdict(lambda: len(self.y_ids))  # 品詞のID
        self.feat_lab = []  # ファイル全体のone-hot-vec
        self.lmd = 0.01  # 学習率
        self.iter = 20  # RNNのループ回数
        self.w_rh = None  # 前の隠れ層からの重み
        self.w_rx = None  # 入力からの重み
        self.w_br = None  # バイアス
        self.w_oh = None  # 出力への重み
        self.w_bo = None  # 出力時のバイアス
        self.w_rh_d = None  # 前の隠れ層からの重みの勾配
        self.w_rx_d = None  # 入力からの重みの勾配
        self.w_br_d = None  # バイアスの勾配
        self.w_oh_d = None  # 出力への重みの勾配
        self.w_bo_d = None  # 出力時のバイアスの勾配
        np.random.seed(seed=1013)

###ここより下でパラメータの設定###

    def initializetion(self, file_name):
        """全体の初期化"""
        train_line=[]
        with open(file_name, "r") as data:
            for line in data:
                train_line.append(line)  # lineを保存
                wordtags = line.strip().split()
                for wordtag in wordtags:
                    word, tag = wordtag.split("_")  # 単語と品詞に分解
                    self.make_id_word(word)  # 単語のIDを設定
                    self.make_id_tag(tag)  # 品詞のIDを設定
            for line in train_line:
                word_tag_vec = self.train_create_features(line)  # 単語とタグのone-hot-vecを生成
                self.feat_lab.append(word_tag_vec)  # 入力ファイルの単語のone-hot-vecの集合を生成
            self.initialize_net()  # 重みとバイアスを初期化

    def make_id_word(self, word):
        """単語のID振り分け"""
        self.x_ids[word]

    def make_id_tag(self, tag):
        """タグのID振り分け"""
        self.y_ids[tag]

    def initialize_net(self):
        """重みとバイアスの初期化"""
        self.w_rx = np.random.rand(self.hidden_num, len(self.x_ids)) / 5 - 0.1
        self.w_rh = np.random.rand(self.hidden_num, self.hidden_num) / 5 - 0.1
        self.w_br = np.random.rand(self.hidden_num) / 5 - 0.1
        self.w_oh = np.random.rand(len(self.y_ids), self.hidden_num) / 5 - 0.1
        self.w_bo = np.random.rand(len(self.y_ids)) / 5 - 0.1

    def train_create_features(self, line):
        """訓練時の特徴量"""
        sent_feat = []
        words = line.strip().split()
        for word in words:
            x, y = word.split("_")  # 単語と品詞を分解
            x_vec = self.create_one_hot(self.x_ids[x], len(self.x_ids))  # 単語のone-hot-vec
            y_vec = self.create_one_hot(self.y_ids[y], len(self.y_ids))  # 品詞のone-hot-vec
            sent_feat.append([x_vec, y_vec])  # 単語と品詞のone-hot-vecのリストをリストに追加
        return sent_feat  # 文のone-hot-vec

    def find_max(self, p):
        """配列の最大値の番号をとってくる"""
        y = 0
        for i in range(len(p)):
            if p[i] > p[y]:
                y = i
        return y

    def create_one_hot(self, id, size):
        """単語のところだけ１が立つような配列の生成"""
        vec = np.zeros(size)
        vec[id] = 1
        return vec

###ここより下で学習###

    def online_learning(self, data):
        """学習"""
        self.initializetion(data)  # 各パラメータを初期化
        for i in tqdm(range(self.iter)):
            for vec in self.feat_lab:
                h, p, y = self.forward_rnn(vec)  # 前に伝搬
                self.gradient_rnn(vec, p, h)  # 逆伝搬
                self.update_weights()  # 重み更新

    def forward_rnn(self, sent):
        """前に伝搬"""
        h = []  # 隠れ層のリスト
        p = []  # 出力の確率
        y = []  # 確率の最大値のタグのID
        for t in range(len(sent)):
            x, _ = sent[t]  # 単語のone-hot-vecをxに代入
            if t > 0:  # １回目以降のループ
                h.append(np.tanh(np.dot(self.w_rx, x) + np.dot(self.w_rh, h[t-1]) + self.w_br))  # 次の隠れ層に進む
            else:  # １回目のループ
                h.append(np.tanh(np.dot(self.w_rx, x) + self.w_br))  # 次の隠れ層に進む（前の隠れ層は取らない）
            p.append(self.softmax(np.dot(self.w_oh, h[t]) + self.w_bo))  # 出力の重みをsoftmaxに通してタグ当たりの確率を生成
            y.append(self.find_max(p[t]))  # 一番確率の高いタグをyに格納
        return h, p, y

    def softmax(self, x):
        """softmaxの生成"""
        u = np.sum(np.exp(x))
        return np.exp(x)/u

    def gradient_rnn(self, vec, p, h):
        """逆伝搬"""
        self.initialize_net_delta()  # 勾配の初期化
        delta_r_d = np.zeros(self.hidden_num)  # エラーの伝搬
        for t in reversed(range(len(vec))):
            x, y_d = vec[t]  
            delta_o_d = y_d - p[t]  # 出力のエラーの計算
            self.w_oh_d += np.outer(delta_o_d, h[t])  # 隠れ層の重みの伝搬
            self.w_bo_d += delta_o_d  # バイアスの伝搬
            delta_r = np.dot(delta_r_d, self.w_rh) + np.dot(delta_o_d, self.w_oh)
            delta_r_d = delta_r*(1 - h[t]**2)  # tanhの微分
            self.w_rx_d += np.outer(delta_r_d, x)
            self.w_br_d += delta_r_d
            if t != 0:
                self.w_rh_d += np.outer(delta_r_d, h[t-1])  # 最初以外は前の隠れ層が入力される

    def initialize_net_delta(self):
        """勾配の初期化"""
        self.w_rx_d = np.zeros((self.hidden_num, len(self.x_ids)))
        self.w_rh_d = np.zeros((self.hidden_num, self.hidden_num))
        self.w_br_d = np.zeros(self.hidden_num)
        self.w_oh_d = np.zeros((len(self.y_ids), self.hidden_num))
        self.w_bo_d = np.zeros(len(self.y_ids))

    def update_weights(self):
        """重みの更新"""
        self.w_rx += self.lmd * self.w_rx_d
        self.w_rh += self.lmd * self.w_rh_d
        self.w_br += self.lmd * self.w_br_d
        self.w_oh += self.lmd * self.w_oh_d
        self.w_bo += self.lmd * self.w_bo_d

###ここより下でテスト###

    def predict(self, data):
        """テスト"""
        with open(data, "r") as f:
            for line in f:
                sent = []
                phi = self.test_create_features(line)
                h, p, y = self.test_forward_rnn(phi)
                for ans in y:
                    for value, id in self.y_ids.items():
                        if id == ans:
                            sent.append(value)
                print(" ".join(sent))

    def test_create_features(self, x):
        """学習データとデータ形式が違うため、テストデータ用の特徴量抽出"""
        phi = []
        words=x.strip().split()
        for word in words:
            if word in self.x_ids:
                phi.append(self.create_one_hot(self.x_ids[word], len(self.x_ids)))
            else:
                phi.append(np.zeros(len(self.x_ids)))
        return phi

    def test_forward_rnn(self, sent):
        """特徴量同様にデータ形式を合わせる"""
        h = []
        p = []
        y = []
        for t in range(len(sent)):
            x = sent[t]  # 品詞がついていないから単語だけ
            if t > 0:
                h.append(np.tanh(np.dot(self.w_rx, x) + np.dot(self.w_rh, h[t-1]) + self.w_br))
            else:
                h.append(np.tanh(np.dot(self.w_rx, x) + self.w_br))
            p.append(self.softmax(np.dot(self.w_oh, h[t]) + self.w_bo))
            y.append(self.find_max(p[t]))
        return h, p, y


if __name__ == "__main__":
    #file_name = "05-train-input.txt"
    file_name = "wiki-en-train.norm_pos"
    rnn = RNN()
    rnn.online_learning(file_name)
    #test_file = "05-test-input.txt"
    test_file = "wiki-en-test.norm"
    rnn.predict(test_file)


"""
hidden_num = 2, iter = 1, lmd = 0.01
Accuracy: 45.43% (2073/4563)

hidden_num = 2, iter = 5, lmd = 0.01
Accuracy: 63.09% (2879/4563)

hidden_num = 2, iter = 10, lmd = 0.01
Accuracy: 72.61% (3313/4563)

hidden_num = 4, iter = 1, lmd = 0.01
Accuracy: 50.34% (2297/4563)

hidden_num = 4, iter = 5, lmd = 0.01
Accuracy: 78.35% (3575/4563)

hidden_num = 4, iter = 10, lmd = 0.01
Accuracy: 83.72% (3820/4563)

hidden_num = 8, iter = 1, lmd = 0.01
Accuracy: 56.54% (2580/4563)

hidden_num = 8, iter = 5, lmd = 0.01
Accuracy: 83.32% (3802/4563)

hidden_num = 8, iter = 10, lmd = 0.01
Accuracy: 86.52% (3948/4563)

hidden_num = 8, iter = 20, lmd = 0.01
Accuracy: 86.57% (3950/4563)

hidden_num = 16, iter = 20, lmd = 0.01
Accuracy: 31.91% (1456/4563)

hidden_num = 16, iter = 20, lmd = 0.001
Accuracy: 77.45% (3534/4563)

hidden_num = 16, iter = 20, lmd = 0.001
Accuracy: 32.92% (1502/4563)
"""