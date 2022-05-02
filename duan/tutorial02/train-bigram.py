import sys
from collections import defaultdict

counts = defaultdict(lambda:0)
context_counts = defaultdict(lambda:0)
#　必要なファイルを読みこむ
training_file = open('./nlptutorial/data/wiki-en-train.word').readlines()

for line in training_file:
    words = line.split()
    words.append('</s>')
    words.insert(0,'<s>')
    for i in range(1,len(words)):
        counts[words[i-1]+' '+words[i]] += 1
        context_counts[words[i-1]] += 1
        counts[words[i]] += 1
        context_counts[''] += 1 

output = open('./nlptutorial/data/train-bigram.txt','w')

for ngram, count in sorted(counts.items()):
    context = ngram.split(' ')
    context.pop()
    context = ' '.join(context)
    probability = float(counts[ngram]/context_counts[context])
    output.write (ngram + '  ' + '{:.6f}'.format(probability) + '\n'
