#python tutorial05.py ../../../nlptutorial/test/03-train-input.txt output.txt   (p.test_perceptron(sys.argv[3], sys.argv[4])はコメントアウトする)
#python tutorial05.py ../../../nlptutorial/data/titles-en-train.labeled predicted_weights.txt ../../../nlptutorial/data/titles-en-test.word predicted_labels.txt
#python2 ../../../nlptutorial/script/grade-prediction.py ../../../nlptutorial/data/titles-en-test.labeled predicted_labels.txt

from collections import defaultdict
import sys

class PerceptronClassify:
    def __init__(self):
        self.weight_dict = defaultdict(lambda: 0) #重み


    def train_online(self, train_file, times): #重みのオンライン学習 　times:訓練データを何周するか
        with open(train_file, "r") as f_train:
            input_data = f_train.readlines()

        for _ in range(times):   
            for line in input_data:
                y_ans, name = line.strip().split("\t") #1行はこういう感じ->（-1 A moat settlement is a village surrounded by a moat .）
                y_ans = int(y_ans)
                phi = self.create_features(name)
                y_pred = self.predict_one(phi)
                #print(y_pred)
                if (y_pred != y_ans):
                    self.update_weights(phi, y_ans)


    def output_weights(self, output_file): #重みをファイルに書き込み
        with open(output_file, "w") as f_out:
            for word, value in sorted(self.weight_dict.items()):
                f_out.write(f'{word} {value:.6f}\n')
            

    def predict_one(self, phi):
        score = 0
        for word, value in phi.items():
            if word in self.weight_dict:
                score += value * self.weight_dict[word]
        if score >= 0:
            return 1
        else:
            return -1


    def create_features(self, x):
        phi = defaultdict(lambda: 0) #phi : xで出てくる単語の頻度を保持してるだけ(UNI:はおまけ)
        words = x.split()
        for word in words:
            phi[f'UNI:{word}'] += 1
        return phi


    def update_weights(self, phi, y):
        for word, value in phi.items():
            self.weight_dict[word] += y * value

    
    def predict_all(self, test_file, output_file): #テストデータの分類
        with open(test_file, "r") as f_test:
            data = f_test.readlines()
        with open(output_file, "w") as f_out:
            for line in data:
                line = line.strip()
                phi = self.create_features(line)
                y_pred = self.predict_one(phi)
                f_out.write(f'{y_pred}\t{line}\n')



if __name__ == "__main__":
    p = PerceptronClassify()
    p.train_online(sys.argv[1], 1) #(訓練データファイル, イテレーション数)
    p.output_weights(sys.argv[2]) #(訓練後の重みを出力するファイル)
    p.predict_all(sys.argv[3], sys.argv[4]) #(テストデータファイル, テストデータの分類結果を出力するファイル)             
