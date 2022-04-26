from collections import defaultdict
import sys
import math

f_model = open('wiki-en-train-output.txt', 'r')
f_test = open('wiki-en-test.word', 'r')
f_output = open('wiki-en-test-output.txt', 'w')
# lambda_1 = 0.95
# lambda_2 = 0.1
v = 1000000
data_model = f_model.readlines()
data_test = f_test.readlines()
model_dic = defaultdict(lambda: 0)
h = 0
w = 0

for line in data_model:
    line = line.strip().split()
    if (len(line) == 2):
        model_dic[line[0]] = float(line[1])
    elif (len(line) == 3):
        model_dic[line[0] + " " + line[1]] = float(line[2])

for lambda_1 in range(10, 100, 20):
    lambda_1 /= 100
    for lambda_2 in range(5, 100, 10):
        lambda_2 /= 100
        for line in data_test:
            line = line.strip().split()
            line.append('</s>')
            line.insert(0, '<s>')
            for i in range(1, len(line)):
                p1 = lambda_1 * model_dic[line[i]] + (1 - lambda_1) / v
                p2 = lambda_2 * model_dic[line[i - 1] + " " + line[i]] + (1 - lambda_2) * p1 
                h += - math.log(p2, 2)
                w += 1
        f_output.write(f'lambda1={lambda_1:.2f} lambda2={lambda_2:.2f}  entropy = {h/w}\n')