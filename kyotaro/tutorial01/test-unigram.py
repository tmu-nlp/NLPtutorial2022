import sys
import math
from collections import defaultdict

def test(train_result, test_data, lam):
    train = []
    test = []

    entropy = 0
    coverage = 0

    for line in train_result:
        line = line.strip()
        word = line.split()
        train.append(word)


    for line in test_data:
        line = line.strip()
        word = line.split()
        word.append("</s>")
        for i in range(len(word)):
            test.append(word[i])

    for i in range(len(test)):
        flag = 0
        for j in range(len(train)):
            if train[j][0] == test[i]:
                h = (-1) * math.log(lam * float(train[j][1]) + (1 - lam) / 1000000, 2)
                entropy += h
                coverage += 1
                flag = 1
        if flag == 0:
            h = (-1) * math.log((1 - lam) / 1000000, 2)
            entropy += h
    
    entropy /= len(test)
    coverage /= len(test)
    return entropy, coverage
    # print(entropy)
    # print(coverage)
    # return 0
    

tra = open(sys.argv[1], "r").readlines()
tes = open(sys.argv[2], "r").readlines()
lam = 0.95

entro, cover = test(tra, tes, lam)

print("entropy = ", entro)
print("coverage = ", cover)