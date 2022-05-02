sents = open('./nlptutorial/data/wiki-en-train.word').readlines()
d = {}
for sent in sents:
    for word in sent.split(' '): 
        if word not in d.keys():
            d[word] = 0
        d[word] += 1
print (sorted(d.items()))
