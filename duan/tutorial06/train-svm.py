import numpy as np
from collections import defaultdict

def train():
    w = defaultdict(lambda:0)
    margin = 50
    c = 0.001
    with open('./nlptutorial/data/titles-en-train.labeled') as f:
        for i, line in enumerate(f):
            phi = defaultdict(lambda:0)
            line = line.strip().split('\t')
            y = int(line[0])
            x = line[1]
            words = x.split(' ')
            for word in words:
                phi[word] += 1
            val = 0
            for word, count in phi.items():
                val += w[word] * count
            val = val * y
            if val <= margin:
                for word, weight in w.items():
                    if abs(weight) < c:
                        w[word] = 0
                    else:
                        w[word] -= np.sign(weight) * c
                for word, count in phi.items():
                    w[word] += count * y
    t = ''
    for word, weight in w.items():
        t += f'{word}\t{round(weight,10)}\n'
    with open('./NLPtutorial2022/duan/tutorial06/model06.txt','w') as f:
        f.write(t)

if __name__ == '__main__':
    train()