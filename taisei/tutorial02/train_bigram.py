from collections import defaultdict
import sys
f = open('wiki-en-train.word', 'r')
f_model = open('wiki-en-train-output.txt', 'w')
data = f.readlines()
bi_dict = defaultdict(lambda: 0)
uni_dict = defaultdict(lambda: 0)
count = 0
for line in data:
    line = line.strip().split()
    line.append('</s>')
    line.insert(0, '<s>')
    for i in range(0, len(line) - 1): #文頭記号<s>は確率1なのでuni_dictには入れない
        uni_dict[line[i]] += 1
        bigram_word = " ".join(line[i:i+2])
        bi_dict[bigram_word] += 1
        count += 1
    uni_dict[line[len(line) - 1]] += 1
    count += 1

for k, v in sorted(bi_dict.items()):
    #f_model.write(k + " " + str('{:.6f}'.format(v / uni_dict[k.split()[0]])) + '\n')
    f_model.write(f'{k} {v / uni_dict[k.split()[0]]}\n')

count -= uni_dict['<s>']
del uni_dict['<s>']

for k, v in sorted(uni_dict.items()):
    #f_model.write(k + " " + str('{:.6f}'.format(v / (count))) + '\n')
    f_model.write(f'{k} {v / count}\n')