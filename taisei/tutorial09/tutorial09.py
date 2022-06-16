from collections import defaultdict
from tqdm import tqdm
import random
import sys
import math

class LDA:
    def __init__(self, num_topics=10, alpha=0.01, beta=0.01):
        random.seed(10)
        self.num_topics = num_topics  # トピックの数
        self.alpha = alpha
        self.beta = beta
        self.xcorpus = []
        self.ycorpus = []
        self.xcounts = defaultdict(lambda: 0)
        self.ycounts = defaultdict(lambda: 0)


    def initialize(self, file):
        with open(file, "r") as f:
            data = f.readlines()
        for docid, line in enumerate(data):
            words = line.strip().split()
            topics = []
            for word in words:
                topic = random.randint(0, self.num_topics-1)
                topics.append(topic)
                self.addcounts(word, topic, docid, 1)
            self.xcorpus.append(words)
            self.ycorpus.append(topics)


    def addcounts(self, word, topic, docid, amount):
        self.xcounts[f'{topic}'] += amount
        self.xcounts[f'{word}|{topic}'] += amount
        self.ycounts[f'{docid}'] += amount
        self.ycounts[f'{topic}|{docid}'] += amount


    def sampleone(self, probs):
        z = sum(probs)
        remaining = random.uniform(0, z)
        for i in range(len(probs)):
            remaining -= probs[i]
            if remaining <= 0:
                return i
                
        print("Eroooor! processing was stopped")
        sys.exit()


    def sampling(self, times):
        for _ in tqdm(range(times)):
            ll = 0
            for i in range(len(self.xcorpus)):
                for j in range(len(self.xcorpus[i])):
                    x = self.xcorpus[i][j]
                    y = self.ycorpus[i][j]
                    self.addcounts(x, y, i, -1)
                    probs = []
                    for k in range(self.num_topics):
                        p_x_k = (self.xcounts[f'{x}|{k}'] + self.alpha) / (self.xcounts[f'{k}'] + self.alpha * len(self.xcorpus))
                        p_k_y = (self.ycounts[f'{k}|{i}'] + self.beta) / (self.ycounts[f'{i}'] + self.beta * len(self.ycorpus))
                        probs.append(p_x_k * p_k_y)
                    new_y = self.sampleone(probs)
                    ll += math.log(probs[new_y])
                    self.addcounts(x, new_y, i, 1)
                    self.ycorpus[i][j] = new_y
            print(ll)


    def test_topic(self, result_file):
        topic_list = []  # topic_list[i]で、トピック[i]の単語たちを保持
        for _ in range(self.num_topics):
            topic_list.append(set())
        """
        topic_list = [set()] * self.num_topics
        だと、topic_list[i]の変更が全ての要素に反映されちゃう
        """
        with open(result_file, "w") as f_result:
            for i in range(len(k.xcorpus)):
                for j in range(len(k.xcorpus[i])):
                    x = k.xcorpus[i][j]
                    y = k.ycorpus[i][j]
                    topic_list[y].add(x)

            for i in range(self.num_topics):
                topic_now = topic_list[i]
                f_result.write(f'topic {i}\n')
                f_result.write(f'{" ".join(sorted(list(topic_now)))}\n')
                f_result.write(f'---------------------------------\n')


if __name__ == "__main__":
    #file_path = '../../../nlptutorial/test/07-train.txt'
    file_path = '../../../nlptutorial/data/wiki-en-documents.word'
    iter = 10
    k = LDA()
    k.initialize(file_path)
    k.sampling(iter)
    k.test_topic('result_wiki.txt')