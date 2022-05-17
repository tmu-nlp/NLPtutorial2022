from collections import defaultdict
import dill

class TestPerceptron:
    def __init__(self):
        self.w = defaultdict(lambda: 0)
    
    def predict_all(self):
        with open("train_result.txt", "r") as train:
            for line in train:
                name, value = line.strip().split()
                self.w[name] = float(value)
        
        with open("titles-en-test.word", "r") as test:
            for x in test:
                x = x.strip()
                phi = self.create_features(x)
                y_p = self.predict_one(phi)
                print(str(y_p) + "\t" + x)
    
    def create_features(self, x):  # 単語分割して素性作成
        phi = defaultdict(lambda: 0)
        words = x.split()
        for word in words:
            phi["UNI:" + word] += 1
        return phi
    
    def predict_one(self, phi):  # 1文で予測
        score = 0
        for name, value in phi.items():
            if name in self.w:
                score += value * int(self.w[name])
        if score >= 0:
            return 1
        else:
            return -1

if __name__ == "__main__":
    test = TestPerceptron()
    test.predict_all()