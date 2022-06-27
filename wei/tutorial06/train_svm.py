'''
l1とマージンで　学習を行うプログラムを実装
学習データ: ../data/titles-en-train.labeled
予測に使うデータ: ../data/titles-en-test.word
'''

from collections import defaultdict
import time

class TrainSvm:
    def __init__(self, c=1e-4, margin=10, iter=1):
        self.weight_dict = defaultdict(lambda: 0)
        self.iter = iter
        self.c = c
        self.margin = margin


    #  p22
    def online_training_margin(self, train_file, model_file):
        with open(train_file, 'r', encoding='utf-8') as f_trian:
            train_data = f_trian.readlines()
        for _ in range(self.iter):
            for line in train_data:
                y, x = line.strip().split('\t')
                phi = self.create_feats(x)
                val = self.get_val(phi, int(y))             # val = w*phi*y
                if val <= self.margin:
                    self.update_weights_l1(phi, int(y))
        with open(model_file, 'w', encoding='utf-8') as f_model:
            for k, v in sorted(self.weight_dict.items()):
                f_model.write(f'{k}\t{v:.6f}\n')

    def predict_all(self, predict_file, res_file):

        with open(predict_file, 'r', encoding='utf-8') as f_pred:
            pred_data = f_pred.readlines()
        with open(res_file, 'w', encoding='utf-8') as f_res:
            for x in pred_data:
                line = x.strip()
                phi = self.create_feats(line)
                y_pred = self.predict_one(phi)

                f_res.write(f'{y_pred}\t{line}\n')

    def create_feats(self, x):
        phi = defaultdict(lambda :0)
        for word in x.split():
            phi[f'UNI:{word}'] += 1
        return phi

    # P30
    def get_val(self, phi, y):
        val = 0
        for word, value in phi.items():
            val += self.weight_dict[word] * value
        val = val * y
        return val

    # p27：オンライン学習でL1正則化
    def update_weights_l1(self, phi, y):
        for name, value in self.weight_dict.items():
            if abs(value) < self.c:
                self.weight_dict[name] = 0
            else:
                if value >= 0:
                    self.weight_dict[name] -= value * self.c
                else:
                    self.weight_dict[name] += value * self.c
            for name, value in phi.items():
                self.weight_dict[name] += value * y

    def predict_one(self, phi):
        score = 0
        for name, value in phi.items():   # score = w * φ(x)
            if name in self.weight_dict.keys():
                score += value * self.weight_dict[name]

        return 1 if score >= 0 else -1


if __name__ == '__main__':
    start = time.time()

    train_file = '../data/titles-en-train.labeled'
    model_out = 'model_output.txt'
    pred_file = '../data/titles-en-test.word'
    results = 'results.txt'

    svm = TrainSvm()
    svm.online_training_margin(train_file, model_out)
    svm.predict_all(pred_file, results)

    end = time.time()
    print(f'time used: {end-start:.4f}')











