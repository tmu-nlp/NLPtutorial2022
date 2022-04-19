from collections import defaultdict
import sys

my_dict = defaultdict(lambda: 0)
my_file = open(sys.argv[1], "r").readlines()
count = 0

for line in my_file:
    line = line.strip().split()

    for word in line:
        my_dict[word] += 1
        count += 1
    my_dict['</s>'] += 1
    count += 1

for my_key, my_value in sorted(my_dict.items()):
    print(my_key, '{:.6f}'.format(1.0 * my_value / count))

