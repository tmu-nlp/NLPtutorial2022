import numpy as np
import random
import math
import sys
from collections import defaultdict
from tqdm import tqdm


class LDA:
    def __init__(self, num_topics, alpha, beta, iter):
        self.xcorpus = []  # 単語が入るリスト
        self.ycorpus = []  # トピックが入るリスト
        self.xcounts = defaultdict(lambda: 0)  # 各単語の数
        self.ycounts = defaultdict(lambda: 0)  # 各トピックの数
        self.num_topics = num_topics  # トピックの種類
        self.iter = iter  # sampleの繰り返し
        self.alpha = alpha
        self.beta = beta
        random.seed(1013)

    def initialize(self, file_name):
        """初期化"""
        with open(file_name, "r") as data:
            for docid, line in enumerate(data):  # 入力文章と文章のIDを取得
                words = line.strip().split()
                topics = []
                for word in words:
                    topic = np.random.randint(
                        self.num_topics - 1)  # 単語のトピックを初期化
                    topics.append(topic)
                    self.addcount(word, topic, docid, 1)  # カウント
                self.xcorpus.append(words)  # 単語コーパスに単語列を追加
                self.ycorpus.append(topics)  # トピックのコーパスにトピックを追加

    def addcount(self, word, topic, docid, amount):
        """条件付き確率を求めるためのカウントを計算"""
        self.xcounts[topic] += amount
        self.xcounts[f'{word}|{topic}'] += amount

        self.ycounts[docid] += amount
        self.ycounts[f'{topic}|{docid}'] += amount

    def sampling(self):
        for _ in tqdm(range(self.iter)):
            for i in range(len(self.xcorpus)):
                ll = 0  # 対数尤度の初期化
                for j in range(len(self.xcorpus[i])):
                    x = self.xcorpus[i][j]  # 単語一つ一つを見る
                    y = self.ycorpus[i][j]  # 対応するトピック
                    self.addcount(x, y, i, -1)  # 見た単語のカウントを減らしていく
                    probs = []
                    for k in range(self.num_topics):
                        p_w_t = self.prob_word_topic(x, k)
                        p_t_d = self.prob_topic_docid(k, i)
                        probs.append(p_w_t * p_t_d)  # トピック確率と単語確率を掛け合わせる　（P19）
                    new_y = self.sampleone(probs)  # 最後の確率のIDを返す？
                    ll += math.log(probs[new_y])  # 対数尤度の計算
                    self.addcount(x, new_y, i, 1)
                    self.ycorpus[i][j] = new_y
            # print(ll)

    def prob_word_topic(self, word, topic):
        """単語確率"""
        Nx = len(self.xcorpus)
        nume = self.xcounts[f'{word}|{topic}'] + self.alpha  # 単語確率の分子
        deno = self.xcounts[topic] + self.alpha * Nx  # 分母
        return nume / deno

    def prob_topic_docid(self, topic, docid):
        """トピック確率"""
        Ny = len(self.ycorpus)
        nume = self.ycounts[f'{topic}|{docid}'] + self.beta  # トピック確率の分子
        deno = self.ycounts[docid] + self.beta * Ny  # トピック確率の分母
        return nume / deno

    def sampleone(self, probs):
        """IDを返す？"""
        z = sum(probs)
        remaining = random.uniform(0, z)
        for i in range(len(probs)):
            remaining -= probs[i]
            if remaining <= 0:
                return i

    def out(self, out_file):
        with open(out_file, "w") as x_out:
            for i in range(len(self.xcorpus)):
                ans = []
                for j in range(len(self.xcorpus[i])):
                    x = self.xcorpus[i][j]
                    y = self.ycorpus[i][j]
                    x_out.write(f'{x}_{y} ')
                x_out.write("\n")

    def out_clean(self, out_file):
        with open(out_file, "w") as x_out:
            topic_words = [set() for _ in range(self.num_topics)]
            for i in range(len(self.xcorpus)):
                for j in range(len(self.xcorpus[i])):
                    x = self.xcorpus[i][j]
                    y = self.ycorpus[i][j]
                    topic_words[y].add(x)

            for i in range(self.num_topics):
                x_out.write(f'topic : {i}\n')
                x_out.write(" ".join(sorted(topic_words[i])))
                x_out.write("\n------------------\n")


if __name__ == "__main__":
    input_file = sys.argv[1]

    num_topics = 5
    alpha = 0.01
    beta = 0.01
    lda = LDA(num_topics, alpha, beta, 20)
    lda.initialize(input_file)
    lda.sampling()
    lda.out("my_ans.txt")
    lda.out_clean("topic_words.txt")
