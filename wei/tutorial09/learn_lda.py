'''
train file: ../data/wiki-en-documents.word
topic_num: 5
'''
from collections import defaultdict
import numpy as np
from tqdm import tqdm

class LDA:
    def __init__(self, topic_num=5):
        self.topic_num =topic_num
        self.xcorpus = []
        self.ycorpus = []
        self.xcounts = defaultdict(lambda: 0)
        self.ycounts = defaultdict(lambda: 0)
        self.vocab_size = set()


    # p22: initialization
    def init(self, input_data):
        with open(input_data, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                doc_id = len(self.xcorpus)     # この文章のIDを獲得
                words = line.strip().split()
                topics = []
                for word in words:
                    topic = np.random.randint(0, self.topic_num)   # randomly select a topic
                    topics.append(topic)
                    self.add_counts(word, topic, doc_id, 1)
                    self.vocab_size.add(word)
                self.xcorpus.append(words)
                self.ycorpus.append(topics)


    # p23: カウントの追加
    def add_counts(self, word, topic, doc_id, amount=1):
        self.xcounts[f'{topic}'] += amount
        self.xcounts[f'{word}|{topic}'] += amount
        self.ycounts[f'{doc_id}'] += amount
        self.ycounts[f'{topic}|{doc_id}'] += amount


    # p24: sampling
    def sampling(self, iter=10, alpha=0.01, beta=0.01):
        for _ in tqdm(range(1, iter+1)):
            lg_hood = 0
            for i in range(len(self.xcorpus)):               # i:doc_id
                for j in range(len(self.xcorpus[i])):        # j:word_id
                    x = self.xcorpus[i][j]
                    y = self.ycorpus[i][j]
                    self.add_counts(x, y, i, amount=-1)
                    probs = []
                    for k in range(self.topic_num):
                        # p10+p21: smoothed Latent Dirichlet Allocation
                        px_k = (self.xcounts[f'{x}|{k}'] + alpha)/(self.xcounts[f'{k}'] + alpha*len(self.vocab_size))
                        pk_y = (self.ycounts[f'{k}|{y}'] + beta)/(self.ycounts[f'{y}'] + beta*self.topic_num)
                        probs.append(px_k * pk_y)
                    new_y = self.sample_one(probs)
                    lg_hood += np.log(probs[new_y])
                    self.add_counts(x, new_y, i, amount=1)
                    self.ycorpus[i][j] = new_y
            print(f'log_likelihood is {lg_hood}.')

        with open('results.txt', 'w', encoding='utf-8') as f_res:
            f_res.write(f'{self.xcounts}\t{self.ycounts}')


    # p14: sample one
    def sample_one(self, probs):
        z = np.sum(probs)                             # 確率の和(正規化項)を計算
        remaining = np.random.uniform(0, z)        # [0,z)の乱数を一様分布によって生成(因为参数size未给出，生成一个标量)
        for i in range(len(probs)):
            remaining -= probs[i]                      # 現在の確率を引く
            if remaining <= 0:
                return i




if __name__ == '__main__':
    train_file = '../data/wiki-en-documents.word'
    topic_lda = LDA()
    topic_lda.init(train_file)
    topic_lda.sampling()

'''
 10%|█         | 1/10 [00:36<05:31, 36.85s/it]log_likelihood is -10322474.516846895.
 20%|██        | 2/10 [01:10<04:40, 35.02s/it]log_likelihood is -10242585.04416713.
 30%|███       | 3/10 [01:42<03:55, 33.62s/it]log_likelihood is -10187169.124501795.
 40%|████      | 4/10 [02:14<03:17, 32.95s/it]log_likelihood is -10143943.816934967.
log_likelihood is -10099291.863565994.
 50%|█████     | 5/10 [02:47<02:45, 33.15s/it]log_likelihood is -10075370.820625762.
 60%|██████    | 6/10 [03:23<02:15, 33.97s/it]log_likelihood is -10045718.425181128.
 70%|███████   | 7/10 [04:00<01:44, 34.86s/it]log_likelihood is -10023708.70028403.
 80%|████████  | 8/10 [04:36<01:10, 35.41s/it]log_likelihood is -9988862.551759424.
100%|██████████| 10/10 [05:53<00:00, 35.31s/it]
log_likelihood is -9966910.8789416.'''










