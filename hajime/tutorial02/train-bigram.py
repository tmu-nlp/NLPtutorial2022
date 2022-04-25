import sys  #i/o
from collections import defaultdict #default-set

counts = defaultdict(lambda: 0)
context_counts = defaultdict(lambda: 0)
trg_file = open(sys.argv[1],"r").readlines()

for line in trg_file:
    line_list = line.strip()
    words = line_list.split(" ")
    words.insert(0,"<s>")
    words.append("</s>")
    for i in range(1,len(words)):
        bigram = " ".join(words[i-1:i+1])
        counts[bigram] += 1
        context_counts[words[i-1]] += 1
        counts[words[i]] += 1
        context_counts[""] += 1

output_file = open("model-file2.txt","w")

for key,value in sorted(counts.items()):
    key_list = key.strip()
    key_words = key_list.split(" ")
    key_words.pop(-1)
    context = " ".join(key_words)
    prob = float(counts[key] / context_counts[context])
    output_file.writelines("%s %r\n" % (key, prob))