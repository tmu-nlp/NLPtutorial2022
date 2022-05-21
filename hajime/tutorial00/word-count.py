#!/usr/bin/python3

import sys  
from collections import defaultdict 

my_dict = defaultdict(lambda: 0)
my_file = open(sys.argv[1],"r").readlines()

for line in my_file:
    line = line.strip()
    words = line.split(" ")
    for word in words:
        my_dict[word] += 1

out_file = open("00-output.txt","w")
    
for foo, bar in sorted(my_dict.items()):
    out_file.writelines("%s %r\n" % (foo, bar))


