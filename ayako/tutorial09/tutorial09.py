from pkg_resources import working_set
from tqdm import tqdm
import random
from collections import defaultdict
from math import log 
import nltk
nltk.download('stopwords')

class LDA:
    def __init__(self, num_topics, alpha, beta, train_file):
        self.xcorpas = [] #各x,yを格納
        self.ycorpas = []
        self.xcounts = defaultdict(lambda:0) #カウントの格納
        self.ycounts = defaultdict(lambda:0)
        self.word_type = defaultdict(lambda:0) #語彙の種類をカウントする用
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.init(train_file)

    #初期化
    def init(self,train_file):
        with open(train_file, "r") as f:
            for doc_id, line in enumerate(f): #doc_id：この文章のID
                words = line.strip().split()
                topics = []
                for word in words:
                    self.word_type[word] #語彙に追加
                    topic = random.randint(0,self.num_topics-1) #[0,num_topics)の乱数でランダムにトピックを初期化
                    topics.append(topic)
                    self.add_counts(word, topic, doc_id, 1) #カウントを追加
                self.xcorpas.append(words)
                self.ycorpas.append(topics)
                
    #カウントの追加
    def add_counts(self, word, topic, doc_id, amount):
        self.xcounts[f"{topic}"] += amount 
        self.xcounts[f"{word}|{topic}"] += amount #そのトピックの時，単語が出てくる回数をカウント
        self.ycounts[f"{doc_id}"] += amount
        self.ycounts[f"{topic}|{doc_id}"] += amount
    
    def sample_one(self, probs):#トピックの分布に従ってサンプルする
        z = sum(probs) #確率の和を計算
        remaining = random.uniform(0,z) #[0,z)の一様分布の乱数を生成
        for i in range(len(probs)): #probsの各項目を検証
            remaining -= probs[i] #現在の確率を引く
            if remaining <= 0: #0より小さい場合，返す
                return i

    #サンプリング
    def sampling(self, max_iter):
        ll = 0 #対数尤度(log likelihood)
        for _ in tqdm(range(max_iter)):
            for i in range(len(self.xcorpas)):
                for j in range(len(self.xcorpas[i])):
                    x = self.xcorpas[i][j]
                    y = self.ycorpas[i][j]
                    self.add_counts(x, y, i, -1)
                    probs = []
                    for k in range(self.num_topics):
                        #トピックkの確率：資料p.23を参考に
                        N_x = len(self.word_type.keys())
                        N_y = self.num_topics
                        p_x_k = ((self.xcounts[f"{x}|{k}"] + self.alpha) / (self.xcounts[f"{k}"] + self.alpha * N_x))
                        p_k_Y = ((self.ycounts[f"{k}|{i}"] + self.beta) / (self.ycounts[f"{i}"] + self.beta * N_y))
                        probs.append(p_x_k * p_k_Y)
                    new_y = self.sample_one(probs)#新しいトピックをサンプルしてくる
                    ll += log(probs[new_y])
                    self.add_counts(x, new_y, i, 1)
                    self.ycorpas[i][j] = new_y
            print(ll)

    def output(self, result_file):
        topics = [set() for _ in range(self.num_topics)] #重複しないようにset型で
        for i in range(len(self.xcorpas)):
            for j in range(len(self.xcorpas[i])):
                x = self.xcorpas[i][j]
                y = self.ycorpas[i][j]
                stopword = nltk.corpus.stopwords.words("english") #ストップワード除去しといた方がいい
                if x in stopword:
                    continue
                topics[y].add(x) #set型の時はaddで要素追加
        
        with open(result_file, "w") as f:
            for i in range(self.num_topics):
                topic = topics[i]
                print(f"---topic #{i}----------------------------------------", file=f)
                print(" ".join(topic), file=f)#出力するときにソートする，記号も削除(&から始まるやつとか)
                #カウントの出力っていらなくない？

if __name__ == "__main__":
    path = "../../../nlptutorial/"
    #テスト用
    train_file1 = path + "test/07-train.txt"
    result1 = "result_test.txt"
    #本番用
    train_file2 = path + "data/wiki-en-documents.word"
    result2 = "result.txt"

    lda = LDA(num_topics=5, alpha=0.01, beta=0.01, train_file=train_file2)
    lda.sampling(max_iter=10)
    lda.output(result_file=result2)

"""各iterationの対数尤度
  0%|                                                                                                                                         | 0/10 [00:00<?, ?it/s]
  -10318934.499361731
 10%|████████████▉                                                                                                                    | 1/10 [00:11<01:42, 11.44s/it]
 -20557295.963465374
 20%|█████████████████████████▊                                                                                                       | 2/10 [00:22<01:30, 11.28s/it]
 -30733619.13140767
 30%|██████████████████████████████████████▋                                                                                          | 3/10 [00:33<01:18, 11.28s/it]
 -40852722.970868066
 40%|███████████████████████████████████████████████████▌                                                                             | 4/10 [00:45<01:08, 11.36s/it]
 -50902931.140146516
 50%|████████████████████████████████████████████████████████████████▌                                                                | 5/10 [00:56<00:56, 11.38s/it]
 -60857822.21836976
 60%|█████████████████████████████████████████████████████████████████████████████▍                                                   | 6/10 [01:08<00:45, 11.39s/it]
 -70686740.64183307
 70%|██████████████████████████████████████████████████████████████████████████████████████████▎                                      | 7/10 [01:19<00:34, 11.37s/it]
 -80373659.22742182
 80%|███████████████████████████████████████████████████████████████████████████████████████████████████████▏                         | 8/10 [01:31<00:22, 11.41s/it]
 -89932432.06486312
 90%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████             | 9/10 [01:42<00:11, 11.58s/it]
 -99395181.43622005
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [01:55<00:00, 11.56s/it]
"""