import numpy as np
from collections import defaultdict

NUM_TOPICS = 5
ITER = 5

class LDA:
    def __init__(self): # 初期化
        # 各x,yを格納する
        self.xcorpus = [] 
        self.ycorpus = [] 
        # カウントの格納
        self.xcounts = defaultdict(int) 
        self.ycounts = defaultdict(int)
        self.init()

    def init(self):
        train_path = './nlptutorial/data/wiki-en-documents.word'
        for line in open(train_path):
            doc_id = len(self.xcorpus) # 文章のIDを獲得する
            # 単語のトピックをランダム初期化する
            topics = [] 
            words = line.split()
            for word in words:
                topic = np.random.randint(0, NUM_TOPICS)
                topics.append(topic)
                self.add_counts(word, topic, doc_id, 1) # カウントを追加する
            self.xcorpus.append(words)
            self.ycorpus.append(topics)

    def add_counts(self, word, topic, doc_id, amount): # カウントの追加
        self.xcounts[f'{topic}'] += amount
        self.xcounts[f'{word}|{topic}'] += amount
        self.ycounts[f'{doc_id}'] += amount
        self.ycounts[f'{topic}|{doc_id}'] += amount

    def sample_one(self, probs):
        z = sum(probs) # 確率の和(正規化項)を計算する
        remaining = np.random.rand() * z # 乱数を一様分布によって生成する
        for i in range(len(probs)): # probsの各項目を検証する
            remaining -= probs[i] # 現在の確率を引く
            if remaining <= 0: # 0より小さい場合、返す
                return i

    def sampling(self): # サンプリング
        for k in range(ITER):
            ll = 0
            for i in range(len(self.xcorpus)):
                for j in range(len(self.xcorpus[i])):
                    x = self.xcorpus[i][j]
                    y = self.ycorpus[i][j]
                    self.add_counts(x, y, i, -1) # 各カウントの減算
                    probs = []
                    for k in range(NUM_TOPICS):
                        p_xk = self.xcounts[f'{x}|{k}'] / (self.xcounts['k'] + 1)
                        p_ky = self.ycounts[f'{k}|{y}'] / (self.ycounts['y'] + 1)
                        probs.append(p_xk * p_ky) # トピックkの確率
                    new_y = self.sample_one(probs)
                    ll += np.log(probs[new_y]) # 対数尤度
                    self.add_counts(x, new_y, i, 1) # 各カウントの加算
                    self.ycorpus[i][j] = new_y
            print(ll)
        for xs, ys in zip(self.xcorpus, self.ycorpus):
            for x, y in zip(xs, ys):
                print(x, y)

if __name__ == '__main__':
    lda = LDA()
    lda.sampling()
