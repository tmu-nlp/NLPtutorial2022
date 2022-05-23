from collections import defaultdict
import re

def create_features(features):
    phi = defaultdict(lambda: 0)
    words = features.split()
    for word in words:
        phi['UNI:' + word] += 1
    return phi

def predict_one(w, phi):
    score = 0
    for name, value in phi.items():
        if name in w:
            score += value + w[name]
    if score >= 0:
        return 1
    else:
        return -1

def predict_all(model_file, input_file):
    w = defaultdict(lambda:0)
    for line in model_file:
        line = line.strip().split('\t')
        name = line[0]
        weight = line[1]
        w[name] = int(weight)
    res = []
    for line in input_file:
        line = line.strip()
        phi = create_features(line)
        y_predict = predict_one(w, phi)
        res.append(str(y_predict) + '\t' + line + '\n')
    return res

if __name__ == '__main__':
    input = open('./nlptutorial/data/titles-en-train.labeled')
    model = open('./NLPtutorial2022/duan/tutorial05/model05.txt')
    ans = open('./NLPtutorial2022/duan/tutorial05/answer.txt','w')
    prediction = predict_all(model, input)
    for line in prediction:
        ans.write(line)
