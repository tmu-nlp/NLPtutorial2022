import sys
from collections import defaultdict


my_file = open(sys.argv[1], "r")
my_dict = defaultdict(lambda : 0)
for line in my_file:
    line = line.strip()
    line = line.split(" ")

    for word in line:
        my_dict[word] += 1

for key , value in sorted(my_dict.items()):
    print(key + ' ' + str(value))



