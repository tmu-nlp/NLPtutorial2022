import sys
from collections import defaultdict

my_file = open(sys.argv[1], "r")

my_dict = defaultdict(lambda: 0)

for line in my_file:
    line = line.strip()
    words = line.split()
    for i in range(len(words)):
        my_dict[words[i]] += 1
    


for key, value in sorted(my_dict.items()):
    print(key + " " + str(value))
