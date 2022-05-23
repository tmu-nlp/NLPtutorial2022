#正則化を効率化したバージョン
#python tutorial06_effici.py ../../../nlptutorial/data/titles-en-train.labeled predicted_weights_effici.txt ../../../nlptutorial/data/titles-en-test.word predicted_labels_effici.txt
#python2 ../../../nlptutorial/script/grade-prediction.py ../../../nlptutorial/data/titles-en-test.labeled predicted_labels_effici.txt
from collections import defaultdict
import argparse
import time

class TrainSvmEffici:
    def __init__(self):
        self.weight_dict = defaultdict(lambda: 0)  # 重み
        self.c = 0.0001  # 正則化係数
        self.margin = 10  # マージン（あるデータの境界面からの距離がこれ以下だと重み更新）
        self.last = defaultdict(lambda: 0)  # 単語が最後に出た時のself.iterの値を保持
        self.iter = 0  # 重みを何回更新しているか（効率化していないときに正則化をしている回数）。効率化のトリックp30 のiterの代用


    def train_online_mergin(self, train_file, times):
        """マージンを利用したオンライン学習"""
        with open(train_file, "r") as f_train:
            train_data = f_train.readlines()

        for _ in range(times):
            for line in train_data:
                y_ans, x = line.strip().split('\t')
                y_ans = int(y_ans)
                phi = self.create_features(x)
                val = self.get_val(phi, y_ans)
                if val <= self.margin:  
                    self.update_weights_l1(phi, y_ans)


    def output_weights(self, output_file): 
        """重みをファイルに書き込み"""
        with open(output_file, "w") as f_out:
            for word, value in sorted(self.weight_dict.items()):
                self.getw(word)  # 書き換える前に正則化（ここでもするべきだと思うけど...）
                f_out.write(f'{word} {value:.6f}\n')
                

    def get_val(self, phi, y_ans):
        """phiの境界面からの距離を求める"""
        val = 0
        for word, value in phi.items():
            val += self.weight_dict[word] * value
        val = val * y_ans 
        return val


    def predict_one(self, phi):
        """phiのクラスを予測"""
        score = 0
        for word, value in phi.items():
            if word in self.weight_dict:
                score += value * self.weight_dict[word]
        if score >= 0:
            return 1
        else:
            return -1


    def create_features(self, x):
        """xの素性を作成"""
        phi = defaultdict(lambda: 0) #phi : xで出てくる単語の頻度を保持してるだけ(UNI:はおまけ)
        words = x.split()
        for word in words:
            phi[f'UNI:{word}'] += 1
        return phi


    def update_weights_l1(self, phi, y):
        """重みを更新　"""
        self.iter += 1  # ここでiterに＋1すべき？s
        for word, value in phi.items():
            self.getw(word)
            self.weight_dict[word] += value * y


    def sign(self, value):
        if value >= 0:
            return 1
        else:
            return -1


    def predict_all(self, test_file, output_file):
        """テストデータの分類"""
        with open(test_file, "r") as f_test:
            data = f_test.readlines()
        with open(output_file, "w") as f_out:
            for line in data:
                line = line.strip()
                phi = self.create_features(line)
                y_pred = self.predict_one(phi)
                f_out.write(f'{y_pred}\t{line}\n')


    def getw(self, name):
        """nameの重みが最後に更新されたとき(last[name])と現在(iter)の差を使って、してなかった分一気に正則化する。"""
        if self.iter != self.last[name]:
            c_size = self.c * (self.iter -self.last[name])
            if abs(self.weight_dict[name]) <= c_size:
                self.weight_dict[name] = 0
            else:
                self.weight_dict[name] -= self.sign(self.weight_dict[name]) * c_size
            self.last[name] = self.iter


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('train_file', help='訓練データファイル')
    parser.add_argument('weight_output_file', help='学習した重みを出力するファイル')
    parser.add_argument('test_file', help='テストデータファイル')
    parser.add_argument('output_file', help='分類結果を出力するファイル')
    args = parser.parse_args()

    k = TrainSvmEffici()
    k.train_online_mergin(args.train_file, 4)
    k.output_weights(args.weight_output_file)
    k.predict_all(args.test_file, args.output_file)

    end_time = time.time()
    print(f'実行時間：{end_time - start_time}')

    """
    iter = 1のとき、実行時間：0.4215359687805176
    iter = 2のと、実行とき間：0.6712310314178467
    iter = 4のとき、実行時間：1.2692830562591553
    iter = 100のとき、実行時間：21.498298168182373

    （参考）正則化を効率化しない場合の実行時間
    iter = 1のとき、実行時間：9.363991975784302
    iter = 2のとき、実行時間：19.583575010299683
    iter = 4のとき、実行時間：36.93076133728027
    """