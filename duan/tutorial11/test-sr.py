# python3 ./nlptutorial/script/grade-dep.py ./nlptutorial/data/mstparser-en-test.dep ./NLPtutorial2022/duan/tutorial11/answer11.txt

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

def ShiftReduce(queue, weights):
    stack = [(0, 'ROOT', 'ROOT')]
    heads = [-1 for i in range(len(queue) + 1)]
    while len(queue) > 0 or len(stack) > 1:
        features = MakeFeatures(stack, queue)
        scores = {}
        scores['SHIFT'] = PredictScore(weights['SHIFT'], features)
        scores['LEFT'] = PredictScore(weights['LEFT'], features)
        scores['RIGHT'] = PredictScore(weights['RIGHT'], features)
        if (max(scores.items(), key=lambda x: x[1])[0] == 'SHIFT' and len(queue) > 0) or len(stack) < 2:
            stack.append(queue.pop(0))
        elif max(scores.items(), key=lambda x: x[1])[0] == 'LEFT':
            heads[stack[-2][0]] = stack[-1][0]
            stack.pop(-2)
        else:
            heads[stack[-1][0]] = stack[-2][0]
            stack.pop(-1)
    return heads

if __name__ == '__main__':
    path_data_test = './nlptutorial/data/mstparser-en-test.dep'
    path_data_out = './NLPtutorial2022/duan/tutorial11/answer11.txt'
    path_weights = './NLPtutorial2022/duan/tutorial11/weights.dump'
    data = MakeData(path_data_test)
    with open(path_weights, 'rb') as data_weights:
        weights = dill.load(data_weights)
    with open(path_data_test) as data_in, open(path_data_out, 'w') as data_out:
        for queue in map(lambda x: x[0], data):
            heads = ShiftReduce(queue, weights)
            for i, line in enumerate(data_in):
                if line == '\n':
                    print(file=data_out)
                    break
                else:
                    print('\t'.join(line.strip().split('\t')[0:6] + [str(heads[i+1])] + [line.strip().split('\t')[-1]]), file=data_out)

# 68.678595% (3186/4639)