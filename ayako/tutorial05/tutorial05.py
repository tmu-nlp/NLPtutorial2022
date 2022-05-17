from collections import defaultdict
import dill

class Perceptron:
    #素性作成
    #後でbigramにする？
    def create_features(self, x):
        phi = defaultdict(lambda : 0)#素性関数のベクトル，単語の出現回数辞書
        words = x.split()
        for word in words:
            phi["UNI:"+word] += 1
        return phi

    #一つの事例に対する予測
    def predict_one(self, w, phi):
        score = 0
        for name, value in phi.items():
            if name in w:
                score += value * w[name]
        if score >= 0:
            return 1
        else:
            return -1

    #重みの更新
    def update_weights(self, w, phi, y):#y=1or-1
        for name, value in phi.items():#既知語だったら重みを更新
            w[name] += value * y
    
    #全事例に対する予測
    def predict_all(self, weights, input_file):
        w = dill.load(open(weights, "rb"))
        with open(input_file, "r") as f:
            for x in f.readlines():
                x = x.strip()
                phi = self.create_features(x)
                y_predict = self.predict_one(w, phi)
                print(str(y_predict) + "\t" + x)

    #パーセプトロンを用いた分類器学習
    def train_perceptron(self, num, model_file, output_file):
        w = defaultdict(lambda : 0)#重みの辞書
        for i in range(num):
            with open(model_file, "r") as model:#ループの外で開くとうまくいかない
                for line in model:
                    y, x = line.strip().split("\t")
                    y = int(y)
                    phi = self.create_features(x)
                    y_predict = self.predict_one(w, phi)
                    if y_predict != y:
                        self.update_weights(w, phi, y)
        dill.dump(w, open(output_file, "wb"))

    def test_perceptron(self, weights, input_file):
        self.predict_all(weights, input_file)

if  __name__ == "__main__":
    path = "../../../nlptutorial/"
    #model_file = path + "test/03-train-input.txt"

    model_file = path + "data/titles-en-train.labeled"
    input_file = path + "data/titles-en-test.word"
    output_file = "weights.dill"#dill使ってみたかった

    p = Perceptron()
    p.train_perceptron(10, model_file, output_file)
    p.test_perceptron(output_file, input_file)

"""
script/grade-predict.pyがpython2で書かれてるから注意
ptintに()をつける

Accuracy = 90.967056%(num = 1)
Accuracy = 91.852639%(num = 5)
Accuracy = 93.446688%(num = 10)
Accuracy = 93.552958%(num = 100)
"""