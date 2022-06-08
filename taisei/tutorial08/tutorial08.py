#python tutorial08.py ../../../nlptutorial/data/wiki-en-train.norm_pos word_id.txt pos_id.txt weight.txt ../../../nlptutorial/data/wiki-en-test.norm result.txt
#perl ../../../nlptutorial/script/gradepos.pl ../../../nlptutorial/data/wiki-en-test.pos result.txt > accuracy.txt
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

np.random.seed(42)

class RNN:
    def __init__(self, alpha=0.01, hidden_node=10, hidden_layer=1):
        self.word_ids = defaultdict(lambda: len(self.word_ids))
        self.pos_ids = defaultdict(lambda: len(self.pos_ids))
        self.feat_lab = []
        self.alpha = alpha  #学習率
        self.w_rh = None
        self.w_rx = None
        self.b_r = None
        self.w_oh = None
        self.b_o = None
        self.hidden_node = hidden_node
        self.hidden_layer = hidden_layer


    def train_rnn(self, train_file, times):
        with open(train_file, "r") as f:
            data_train = f.readlines()

        self.make_ids(data_train)
        for line in data_train:
            self.make_feature_and_ans(line)

        self.initialize_network()
        for _ in tqdm(range(times)):
            for words, poses in self.feat_lab:
                #print(words, poses)
                h, p, y = self.forward_rnn(words)
                diff_w_rx, diff_w_rh, diff_b_r, diff_w_oh, diff_b_o = self.gradient_rnn(words, h, p, poses) #posesは正解ラベルを渡す
                self.update_weights(diff_w_rx, diff_w_rh, diff_b_r, diff_w_oh, diff_b_o)
            

    def make_ids(self, data_train):
        """訓練データに出現する単語と品詞にIDをつける"""
        for line in data_train:
            line = line.strip().split()
            for word_pos in line:
                word, pos = word_pos.split('_') #訓練データの形式 → Natural_JJ language_NN processing_NN
                self.word_ids[word]
                self.pos_ids[pos]


    def make_feature_and_ans(self, line):
        """feat_labにappendするものをlineごとに作成"""
        #各行の単語をonehot表現した配列　と　品詞をonehot表現した配列をfeat_labに追加
        # feat_lab[i] は　訓練データi+1行目で、[0]は単語のonehot表現の配列、[1]は品詞のonehot表現の配列
        line = line.strip().split()
        words = [] #lineの単語のonehotの配列を格納してく　I eat an apple だったらwords[2]はanのonehotベクトル
        poses = []
        for word_pos in line:
            word, pos = word_pos.split('_')
            word_vec = self.create_one_hot(self.word_ids[word], len(self.word_ids)) #wordをonehotで表現
            pos_vec = self.create_one_hot(self.pos_ids[pos], len(self.pos_ids)) #posをonehotで表現
            words.append(word_vec)
            poses.append(pos_vec)
        self.feat_lab.append([words, poses])


    def initialize_network(self):
        """重みの初期化"""
        self.w_rx = np.random.rand(self.hidden_node, len(self.word_ids)) / 5 - 0.1
        self.w_rh = np.random.rand(self.hidden_node, self.hidden_node) / 5 - 0.1
        self.b_r = np.random.rand(self.hidden_node) / 5 - 0.1
        self.w_oh = np.random.rand(len(self.pos_ids), self.hidden_node) / 5 - 0.1
        self.b_o = np.random.rand(len(self.pos_ids)) / 5 - 0.1


    def initialize_network_diff(self):
        """重みの更新分を求める変数たちを0で初期化"""
        diff_w_rx = np.zeros((self.hidden_node, len(self.word_ids)))
        diff_w_rh = np.zeros((self.hidden_node, self.hidden_node))
        diff_b_r = np.zeros(self.hidden_node)
        diff_w_oh = np.zeros((len(self.pos_ids), self.hidden_node))
        diff_b_o = np.zeros(len(self.pos_ids))
        return diff_w_rx, diff_w_rh, diff_b_r, diff_w_oh, diff_b_o


    def find_best(self, p):
        """配列pで最大値をとるインデックスを返す"""
        return np.argmax(p)


    def create_one_hot(self, id, size):
        """インデックス=idのみが1でサイズ=sizeのone-hotベクトルを返す"""
        vec = np.zeros(size)
        vec[id] = 1
        return vec


    def forward_rnn(self, x):
        """前向き計算"""
        h = [0] * len(x)  # 隠れ層の値
        p = [0] * len(x)  # p[i]：単語[i]に対する品詞の確率分布の値。各単語に対して、len(self.pos_ids)の長さの配列が格納される
        y = [0] * len(x)  # 品詞の予測結果(posのid)
        for t in range(len(x)):
            if t > 0:
                h[t] = np.tanh(np.dot(self.w_rx, x[t]) + np.dot(self.w_rh, h[t-1]) + self.b_r)
            else:
                h[t] = np.tanh(np.dot(self.w_rx, x[t]) + self.b_r)
            p[t] = np.tanh(np.dot(self.w_oh, h[t]) + self.b_o)
            y[t] = self.find_best(p[t])
            #print(p[t])
            
        return h, p, y


    def gradient_rnn(self, x, h, p, y_ans):
        """完全勾配計算"""
        diff_w_rx, diff_w_rh, diff_b_r, diff_w_oh, diff_b_o = self.initialize_network_diff()
        delta_r_dash = np.zeros(self.hidden_node)
        for t in range(len(x))[::-1]:
            # delta_o_dash：出力した予測品詞の確率のエラー。　diff_w_oh, diff_b_o：出力層の重みの勾配。
            delta_o_dash = y_ans[t] - p[t] #y_ans[t]：経験分布（正解の品詞のIDのみ1でそれ以外は0）　p[t]：予測分布
            diff_w_oh += np.outer(delta_o_dash, h[t]) #出力層の重みの勾配 (derr/dwiの式的な（7章 P32））
            diff_b_o += delta_o_dash

            # delta_r：逆伝搬してきたやつ。多分derr/dhtみたいな感じだと思う
            delta_r = np.dot(delta_r_dash, self.w_rh) + np.dot(delta_o_dash, self.w_oh)
            delta_r_dash = delta_r * (1 - h[t] * h[t])
            diff_w_rx += np.outer(delta_r_dash, x[t])
            diff_b_r += delta_r_dash
            if t != 0:
                diff_w_rh += np.outer(delta_r_dash, h[t-1])

        return diff_w_rx, diff_w_rh, diff_b_r, diff_w_oh, diff_b_o
        """
            重みの勾配のイメージ
            a ---q---> c
            (重み行列qで繋がる、入力するベクトルaと出力されるベクトルc)
            重み行列 q 各要素の勾配はnp.outer(c, a)の各要素としてで求められるっぽい
         """


    def update_weights(self, diff_w_rx, diff_w_rh, diff_b_r, diff_w_oh, diff_b_o):
        """重み更新"""
        self.w_rx += self.alpha * diff_w_rx
        self.w_rh += self.alpha * diff_w_rh
        self.b_r += self.alpha * diff_b_r
        self.w_oh += self.alpha * diff_w_oh
        self.b_o += self.alpha * diff_b_o


    def output_weights(self, output_file):
        """ファイルに重みを書き込む"""
        with open(output_file, "w") as f_out:
            f_out.write(f'w_rx\n{self.w_rx}\nw_rh\n{self.w_rh}\nb_r\n{self.b_r}\nw_oh\n{self.w_oh}\nb_o\n{self.b_o}\n')

    
    def output_ids_word(self, output_file):
        """ファイルに単語とそのIDを書き込む"""
        with open(output_file, "w") as f_out:
            for word, id in self.word_ids.items():
                f_out.write(f'{word}\t{id}\n')


    def output_ids_pos(self, output_file):
        """ファイルに品詞とそのIDを書き込む"""
        with open(output_file, "w") as f_out:
            for pos, id in self.pos_ids.items():
                f_out.write(f'{pos}\t{id}\n')


    def predict_all(self, test_file, result_file):
        """テストデータに適用"""
        with open(test_file, "r") as f_test:
            data_test = f_test.readlines()
        with open(result_file, "w") as f_result:
            for line in data_test:
                words = line.strip().split()
                words_vec = []
                
                for word in words:
                    if word in self.word_ids.keys():
                        word_vec = self.create_one_hot(self.word_ids[word], len(self.word_ids))
                    else:  # 訓練時にでてこなかったなら全部0にしなきゃ
                        word_vec = np.zeros(len(self.word_ids))
                    words_vec.append(word_vec)

                y_pred = self.predict_one(words_vec)
                for i, pred in enumerate(y_pred):
                    for pos, id in self.pos_ids.items():
                        if pred == id:
                            y_pred[i] = pos
                            break
                f_result.write(f'{" ".join(y_pred)}\n') 


    def predict_one(self, words_vec):
        h, p, y = self.forward_rnn(words_vec)
        return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file", help="訓練データファイル")
    parser.add_argument("id_word_file", help="訓練データ内で出現する単語に割り当てたIDを出力するファイル")
    parser.add_argument("id_pos_file", help="訓練データ内で出現する品詞に割り当てたIDを出力するファイル")
    parser.add_argument("weight_file", help="学習した重みを出力するファイル")
    parser.add_argument("test_file", help="テストデータファイル")
    parser.add_argument("result_file", help="予測品詞を出力するファイル")
    args = parser.parse_args()
    
    iter = 50
    k = RNN(0.01, 80, 1)
    k.train_rnn(args.train_file, iter)
    k.output_ids_word(args.id_word_file)
    k.output_ids_pos(args.id_pos_file)
    k.output_weights(args.weight_file)
    k.predict_all(args.test_file, args.result_file)


