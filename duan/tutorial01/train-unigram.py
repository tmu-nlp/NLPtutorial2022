from collections import defaultdict

counts = defaultdict(lambda:0)
total_count = 0
training_file = open('./nlptutorial/data/wiki-en-train.word').readlines()

for line in training_file:
    words = line.split()
    words.append('</s>')
    for word in words:
        counts[word] += 1
        total_count += 1 

output = open('./nlptutorial/data/train-unigram.txt','w')

for word, count in sorted(counts.items()):
    probability = counts[word]/total_count
    output.write (word + ' ' + '{:.6f}'.format(probability) + '\n')
