import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def create_features(sentence, ids): # 素性の作成
	phi = np.zeros(len(ids)) # ゼロで初期化する
	words = sentence.split()
	for word in words:
		phi[ids['UNI:'+word]] += 1
	return phi

def init_network(feature_size, layer, node): # ネットワークをランダムな値で初期化する
    # 1つ目の隠れ層
	w0 = np.random.rand(node, feature_size) / 5 - 0.1
	b0 = np.random.rand(1, node) / 5 - 0.1
	net = [(w0, b0)]
    # 中間層
	while len(net) < layer:
		w = np.random.rand(node, node) / 5 - 0.1
		b = np.random.rand(1, node) / 5 - 0.1
		net.append((w, b))
    # 出力層
	w_o = np.random.rand(1, node) / 5 - 0.1
	b_o = np.random.rand(1, 1) / 5 - 0.1
	net.append((w_o, b_o))
	return net

def forward_nn(net, phi0): # ニューラルネットの伝搬
    # 各層の値を取得する
	phi = [0 for i in range(len(net) + 1)]
	phi[0] = phi0
	for i in range(len(net)):
		w, b = net[i]
        # 前の層の値に基づいて値を計算する
		phi[i + 1] = np.tanh(np.dot(w, phi[i]) + b).T
	return phi # 各層の結果を返す

def backward_nn(net, phi, label): # ニューラルネットの逆伝搬
	J = len(net)
    # 各層の誤差
	delta = np.zeros(J + 1, dtype=np.ndarray)
    # 出力層の誤差
	delta[-1] = np.array([label - phi[J][0]])
    # tanhの勾配を考慮した誤差
	delta_p = np.zeros(J + 1, dtype=np.ndarray)
	for i in range(J, 0, -1):
		delta_p[i] = (1 - np.square(phi[i])).T * delta[i]
		w, j = net[i - 1]
		delta[i - 1] = np.dot(delta_p[i], w)
	return delta_p

def update_weights(net, phi, delta, lambda_w): # 重みの更新
	for i in range(len(net)):
		w, b = net[i]
		w += lambda_w * np.outer(delta[i + 1], phi[i])
		b += lambda_w * delta[i + 1]

def train_nn(train_file, output_file, lambda_w=0.03, epoch=30, hidden_l=1, hidden_n=2):
	ids = defaultdict(lambda: len(ids)) # 素性を整数IDに変える
	feat_lab = []

    # 素性を作る
	for line in open(train_file):
		s, sentence = line.strip().split('\t')
		for word in sentence.split():
			ids['UNI:' + word]
	for line in open(train_file):
		label, sentence = line.strip().split('\t')
		label = int(label)
		phi = create_features(sentence, ids)
		feat_lab.append((phi, label))

    # ネットワークの初期化
	net = init_network(len(ids), hidden_l, hidden_n)

    # 学習を行う
	for i in range(epoch): 
		for phi0, label in feat_lab:
			phi = forward_nn(net, phi0)
			delta = backward_nn(net, phi, label)
			update_weights(net, phi, delta, lambda_w)
	with open(output_file, 'wb') as f: # ファイルに保存する
		pickle.dump(net, f)
		pickle.dump(dict(ids), f)

if __name__ == '__main__':
	train_file = './nlptutorial/data/titles-en-train.labeled'
	model_file = './NLPtutorial2022/duan/tutorial07/model07.txt'
	train_nn(train_file, model_file)