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

def update_weights(w, phi, y):
    for name, value in phi.items():
        w[name] += int(value) * int(y)

if __name__ == "__main__":
    weight = defaultdict(lambda:0)
    text = open('./nlptutorial/data/titles-en-train.labeled')
    
    for line in text:
        data = line.strip().split('\t')
        ftrs = data[1]
        label = data[0]
        phi = create_features(ftrs)
        y_predict = predict_one(weight, phi)
        if y_predict != label:
            update_weights(weight, phi, label)
    
    model_out = open('./NLPtutorial2022/duan/tutorial05/model05.txt','w')
    for name, wgt in weight.items():
        model_out.write(name + '\t' + str(round(wgt,6)) + '\n')
