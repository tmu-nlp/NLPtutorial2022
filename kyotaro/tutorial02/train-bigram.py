import sys
from collections import defaultdict

my_file = open(sys.argv[1], "r").readlines()

counts = defaultdict(lambda: 0)
context_counts = defaultdict(lambda: 0)

with open(sys.argv[1], "r") as my_file:
    for line in my_file:
        line = line.strip()
        words = line.split()
        words.append("</s>")
        words.insert(0, "<s>")
        for i in range(1, len(words)):
            bi = " ".join(words[i-1:i+1])
            counts[bi] += 1
            context_counts[words[i - 1]] += 1
            counts[words[i]] += 1
            context_counts[""] += 1




with open(sys.argv[2], "w") as out_file:
    for ngram, count in sorted(counts.items()):

        ngram = ngram.split()
        if len(ngram) >= 2:
            for pre, mother in sorted(context_counts.items()):
                if ngram[0] == pre:
                    #ans.append(count / mother)
                    out_file.write(" ".join(ngram) + " " + str('{:.6f}'.format(count / mother)))
                    out_file.write("\n")
        if len(ngram) == 1:
            p = count / context_counts['']
            out_file.write(" ".join(ngram) + " " + str('{:.6f}'.format(p)))
            out_file.write("\n")
            