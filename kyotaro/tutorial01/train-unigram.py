import sys
from collections import defaultdict

def train(my_dict):
    my_file = open(sys.argv[1], "r").readlines()
    p = []
    ans = 1
    count = 0

    for line in my_file:
        line = line.strip()
        word = line.split()
        word.append("</s>")
        count += len(word)
        for i in range(len(word)):
            my_dict[word[i]] += 1

    for key, value in sorted(my_dict.items()):
        p.append(value / count)

    for i in range(len(p)):
        ans *= p[i]
    
    index = 0
    for key, value in sorted(my_dict.items()):
        print(key + " " + str('{:.6f}'.format(p[index])))
        index += 1
        
my_dict = defaultdict(lambda: 0)
train(my_dict)