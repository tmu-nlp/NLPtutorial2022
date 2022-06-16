from collections import defaultdict
import numpy as np
import random
from nltk.corpus import stopwords


class LDA():
    def __init__(self, num_topics, iter_num, file):
        self.xcorpus = []
        self.ycorpus = []
        self.xcount = defaultdict(lambda: 0)
        self.ycount = defaultdict(lambda: 0)
        self.num_topics = num_topics
        self.word_type = set()
        self.iter_num = iter_num
        self.alpha = 0.01
        self.beta = 0.01
        self.initialize(file)
        self.sampling()

    def addcounts(self, word, topic, docid, amount):
        self.xcount[topic] += amount
        self.xcount[f"{word}|{topic}"] += amount
        self.xcount[docid] += amount
        self.xcount[f"{topic}|{docid}"] += amount

    def initialize(self, file):
        with open(file, "r") as i_file:
            for line in i_file:
                docid = len(self.xcorpus)
                words = line.strip().split()
                topics = []
                for word in words:
                    topic = np.random.randint(self.num_topics)
                    topics.append(topic)
                    self.addcounts(word, topic, docid, 1)
                    self.word_type.add(word)
                self.xcorpus.append(words)
                self.ycorpus.append(topics)

    def sampleone(self, probs):
        z = sum(probs)
        remaining = random.uniform(0, z)
        for i in range(len(probs)):
            remaining -= probs[i]
            if remaining <= 0:
                return i

    def sampling(self):
        for _ in range(self.iter_num):
            logl = 0
            for i in range(len(self.xcorpus)):
                for j in range(len(self.xcorpus[i])):
                    x = self.xcorpus[i][j]
                    y = self.ycorpus[i][j]
                    self.addcounts(x, y, i, -1)
                    probs = []
                    for k in range(self.num_topics):
                        Px_k = (self.xcount[f"{x}|{k}"] + self.alpha)/(
                            self.xcount[f"{k}"] + self.alpha * len(self.word_type))
                        Pk_y = (self.ycount[f"{k}|{y}"] + self.alpha) / \
                            (self.ycount[f"{y}"] + self.beta * self.num_topics)
                        probs.append(Px_k * Pk_y)
                    new_y = self.sampleone(probs)
                    logl += np.log(probs[new_y])
                    self.addcounts(x, new_y, i, 1)
                    self.ycorpus[i][j] = new_y
            print(logl)

    def topic_output(self):
        topic_list = [set() for _ in range(self.num_topics)]

        with open("09answer.txt", "w") as o_file:
            for i in range(len((self.xcorpus))):
                for j in range(len(self.xcorpus[i])):
                    x = self.xcorpus[i][j]
                    y = self.ycorpus[i][j]
                    stopword = stopwords.words("english")
                    if x in stopword:
                        continue
                    topic_list[y].add(x)

            for i in range(self.num_topics):
                topic_now = topic_list[i]
                o_file.write(f'topic {i}\n')
                o_file.write(f'{" ".join(sorted(list(topic_now)))}\n')
                o_file.write(f'---------------------------------\n')

        # for i in range(len(self.xcorpus)):
        #     print(self.xcorpus[i])
        # print("---------------")
        # for i in range(len(self.ycorpus)):
        #     print(self.xcorpus[i])


if __name__ == "__main__":
    # input_file = "test/07-train.txt"
    # lda1 = LDA(2, 10, input_file)
    # lda1.topic_output()

    input_file = "data/wiki-en-documents.word"
    lda2 = LDA(5, 20, input_file)
    lda2.topic_output()
