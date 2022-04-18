import sys  #i/o
from collections import defaultdict #default-set

# 余力があれば関数やクラスにしてみる

trg_file = open(sys.argv[1],"r").readlines()
counts = defaultdict(lambda: 0)
total_count = 0

for line in trg_file:
    line_list = line.strip()
    words = line_list.split(" ")
    words.append("</s>")
    for word in words:
        counts[word] += 1
        total_count += 1

output_file = open("model-file.txt","w")


for key,value in sorted(counts.items()):
    prob = float(counts[key] / total_count)
    output_file.writelines("%s %r\n" % (key, prob))
