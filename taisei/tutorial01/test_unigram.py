import sys
from collections import defaultdict
import math

lambda_1 = 0.95
lambda_unk = 1 - lambda_1
v = 1000000

def calcu_h(model_dict, test_dict):
    p_sum = 0
    for test_key, test_value in test_dict.items():
        if (test_key in model_dict.keys()):
            p = lambda_1 * model_dict[test_key] + lambda_unk / v
        else:
            p = lambda_unk / v
        p_sum += - test_value * math.log(p, 2)
    return p_sum / sum(test_dict.values())


def calcu_coverage(model_dict, test_dict):
    count = 0
    for test_key, test_value in test_dict.items():
        if (test_key in model_dict.keys()): #辞書.keysでその辞書のキーのリストが返される
            count += test_value
    return 1.0 * count / sum(test_dict.values())


model_file = open(sys.argv[1]).readlines()
test_file = open(sys.argv[2]).readlines()

model_dict = defaultdict(lambda: 0)
test_dict = defaultdict(lambda: 0)

for line in model_file:
    line = line.strip().split()
    model_dict[line[0]] = float(line[1]) # model_dictのkeyは単語、valueは確率

for line in test_file:
    line = line.strip().split()
    for word in line:
        test_dict[word] += 1
    test_dict['</s>'] += 1

print('entropy =', '{:.6f}'.format(calcu_h(model_dict, test_dict)))
print('coverage =', '{:.6f}'.format(calcu_coverage(model_dict, test_dict)))
