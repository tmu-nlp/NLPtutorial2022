from collections import defaultdict
'''
a classification ML algorithm to linearly classify given data in two parts
 using decision boundary(a line in 2D or 3D in plane) defined as f(x, θ)=0
 -> Discriminative model, others like linear regression, logistic regression, LDA, SVM, NN, Guassion process, CRF
 (Generative model includes naive bayes, KNN, HMM,Bayes network, MRF)
 y_i(θx+b) >0 or < 0 equivalent to sign() defined in function predict_one() here'''

class Perceptron():
    # model online learning
    def train_perceptron(self, model_file, output_file, iter):
        w = defaultdict(int)           # {w['uni-gram==word'] : count(w),...}のように格納
        with open(model_file, 'r', encoding='utf-8') as m_file:
            m_file = m_file.readlines()

        for i in range(1, iter+1):

            for line in m_file:
                y, x = line.strip().split('\t')      # training data
                y = int(y)
                phi = self.create_features(x)         # extract features from raw text, feature: n-gram tfidf
                y_pred = self.predict_one(w, phi)

                if y_pred != y:
                    self.update_weights(w, phi, y)

        with open(output_file, 'w', encoding='utf-8') as f:
            for k,v in w.items():
                f.write(k + '\t' + str(v) + '\n')


    # 1-gram素性を作成
    def create_features(self, x):
        phi = defaultdict(int)
        words = x.split()
        for word in words:
            phi['UNI:'+word] += 1   # 各1-gram素性の出現頻度を統計
        return phi


    # 1つの事例に対する予測==sign(w*φ(x))を計算して分類
    def predict_one(self, w, phi):
        score = 0                   # score = w*φ(x)
        for k, v in phi.items():
            if k in w:
                score += v * w[k]
        if score >= 0:
            return 1
        else:
            return -1

    def update_weights(self, w, phi, y):
        # y=1の場合、素性の重みを増やす；-1の場合、減らす
        for k, v in phi.items():
            w[k] += int(v*y)


    # test-perceptron
    def predict_all(self, model_out, test_file):
        # load w from model_out
        w = defaultdict(int)
        with open(model_out, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split()    # list like ['UNI:a','2300']
                w[line[0]] = int(line[1])

        with open(test_file, 'r', encoding='utf-8') as f,\
                open('./pred_result.txt', 'w', encoding='utf-8') as f_out:
            for line in f.readlines():
                phi = self.create_features(line)
                y_pred = self.predict_one(w, phi)
                f_out.write(str(y_pred) + '\t' + line)




if __name__ == '__main__':
    percep= Perceptron()
    percep.train_perceptron('../data/titles-en-train.labeled','./model-out.txt', 10)
    percep.predict_all('./model-out.txt','../data/titles-en-test.word')
















