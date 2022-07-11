import copy
import dill
import random
from collections import defaultdict

def MakeData(path_data_in):
    data = list()
    queue = list()
    heads = [-1]
    with open(path_data_in) as data_in:
        for line in data_in:
            if line == '\n':
                data.append((queue, heads))
                queue = list()
                heads = [-1]
            else:
                labels = line.strip().split('\t')
                ID, word, pos, head = int(labels[0]), labels[1], labels[3], int(labels[6])
                queue.append((ID, word, pos))
                heads.append(head)
    return data

def MakeFeatures(stack, queue):
    features = defaultdict(lambda: 0)
    if len(stack) > 0 and len(queue) > 0:
        features['W-1' + stack[-1][1] + 'W0' + queue[0][1]] += 1
        features['W-1' + stack[-1][1] + 'P0' + queue[0][2]] += 1
        features['P-1' + stack[-1][2] + 'W0' + queue[0][1]] += 1
        features['P-1' + stack[-1][2] + 'P0' + queue[0][2]] += 1
    if len(stack) > 1:
        features['W-2' + stack[-2][1] + 'W-1' + stack[-1][1]] += 1
        features['W-2' + stack[-2][1] + 'P-1' + stack[-1][2]] += 1
        features['P-2' + stack[-2][2] + 'W-1' + stack[-1][1]] += 1
        features['P-2' + stack[-2][2] + 'P-1' + stack[-1][2]] += 1
    return features

def PredictScore(weight, features):
    score = 0
    for key, value in features.items():
        if key in weight.keys():
            score += value*weight[key]
    return score

def Update_Weight(weights, features, predict, correct):
    for key in features.keys():
        weights[predict][key] -= features[key]
        weights[correct][key] += features[key]

def ShiftReduceTrain(queue, heads, weights):
    stack = [(0, 'ROOT', 'ROOT')]
    unproc = list()
    for i in range(len(heads)):
        unproc.append(heads.count(i))
    while len(queue) > 0 or len(stack) > 1:
        features = MakeFeatures(stack, queue)
        scores = {}
        scores['SHIFT'] = PredictScore(weights['SHIFT'], features)
        scores['LEFT'] = PredictScore(weights['LEFT'], features)
        scores['RIGHT'] = PredictScore(weights['RIGHT'], features)
        if (max(scores.items(), key=lambda x: x[1])[0] == 'SHIFT' and len(queue) > 0) or len(stack) < 2:
            predict = 'SHIFT'
        elif max(scores.items(), key=lambda x: x[1])[0] == 'LEFT':
            predict = 'LEFT'
        else:
            predict = 'RIGHT'
        if len(stack) < 2:
            correct = 'SHIFT'
        elif heads[stack[-2][0]] == stack[-1][0] and unproc[stack[-2][0]] == 0:
            correct = 'LEFT'
        elif heads[stack[-1][0]] == stack[-2][0] and unproc[stack[-1][0]] == 0:
            correct = 'RIGHT'
        else:
            correct = 'SHIFT'
        if predict != correct:
            Update_Weight(weights, features, predict, correct)
        if correct == 'SHIFT':
            stack.append(queue.pop(0))
        elif correct == 'LEFT':
            unproc[stack[-1][0]] -= 1
            stack.pop(-2)
        elif correct == 'RIGHT':
            unproc[stack[-2][0]] -= 1
            stack.pop(-1)

if __name__ == '__main__':
    path_data_train = './nlptutorial/data/mstparser-en-train.dep'
    path_weights = './NLPtutorial2022/duan/tutorial11/weights.dump'
    n_epoch = 12
    data = MakeData(path_data_train)
    weights = dict()
    weights['SHIFT'] = defaultdict(lambda: 0)
    weights['LEFT'] = defaultdict(lambda: 0)
    weights['RIGHT'] = defaultdict(lambda: 0)
    for epoch in range(n_epoch):
        data_ = copy.deepcopy(data)
        random.seed(epoch)
        random.shuffle(data_)
        for queue, heads in data_:
            ShiftReduceTrain(queue, heads, weights)
    with open(path_weights, 'wb') as data_weights:
        dill.dump(weights, data_weights)