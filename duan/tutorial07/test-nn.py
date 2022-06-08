# python3 ./nlptutorial/script/grade-prediction.py ./nlptutorial/data/titles-en-test.labeled ./NLPtutorial2022/duan/tutorial07/answer07.txt

import pickle
import numpy as np
from collections import defaultdict

def create_features_test(sentece, ids): # 素性を作成する
	phi = np.zeros(len(ids))
	words = sentece.split()
	for word in words:
		if 'UNI:' + word not in ids:
			continue
		phi[ids['UNI:' + word]] += 1
	return phi

def predict_one(net, phi0): # 一つの事例を予測する
	phi = [0 for i in range(len(net) + 1)]
	phi[0] = phi0
	for i in range(len(net)):
		w, b = net[i]
		phi[i + 1] = np.tanh(np.dot(w, phi[i]) + b).T
	score = phi[len(net)][0]
	return 1 if score >= 0 else -1

def test_nn(test_file, model_file, output_file):
	with open(model_file, 'rb') as f: # モデルの読み込み
		net = pickle.load(f)
		ids = pickle.load(f)
	with open(output_file, 'w') as f: # 結果の出力
		for line in open(test_file):
			sentence = line.strip()
			phi = create_features_test(sentence, ids)
			prediction = predict_one(net, phi)
			f.write(f'{prediction}\t{sentence}\n')

if __name__ == '__main__':
	model_file = './NLPtutorial2022/duan/tutorial07/model07.txt'
	test_file = './nlptutorial/data/titles-en-test.word'
	test_nn(test_file, model_file, './NLPtutorial2022/duan/tutorial07/answer07.txt')

# Accuracy = 94.721927%