import math
unigram_prob = open('./nlptutorial/test/04-model.txt').readlines()
unigram = {}
for line in unigram_prob:
    prob = line.strip().split()
    unigram[prob[0]] = prob[1]
    
input = open('./nlptutorial/test/04-input.txt').readlines()

for line in input:
    line = line.strip()
    best_edge = {}
    best_score = {}
    best_edge[0] = None
    best_score[0] = 0
    for word_end in range(1, len(line)+1):
        best_score[word_end] = 10**10
        for word_begin in range(0, word_end):
            word = line[word_begin : word_end]
            if word in unigram.keys() or len(word) == 1:
                prob = float(unigram[word])
                my_score = float(best_score[word_begin]) - math.log(prob)
                if my_score < best_score[word_end]:
                    best_score[word_end] = my_score
                    best_edge[word_end] = (word_begin, word_end)
    words = []
    next_edge = best_edge[len(best_edge)-1]
    while next_edge != None:
        word = line[next_edge[0]:next_edge[1]]
        words.append(word)
        next_edge = best_edge[next_edge[0]]
    words.reverse()
    print(' '.join(words))