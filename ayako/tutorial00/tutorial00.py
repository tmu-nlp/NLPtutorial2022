import sys
from collections import defaultdict

my_dict = defaultdict(lambda: 0)#デフォルトの値を0にする
my_file = open(sys.argv[1], "r").readlines()#コマンドラインで指定したファイルを読み込む


for line in my_file:
    my_list = line.strip().split()#改行除いて配列に格納
    
    for value in my_list:
        my_dict[value] += 1#defaultdictなら存在しないキーの時は0にしてくれる

for key, value in sorted(my_dict.items()):##キー・値の各組をキー順に表示
    print(key,value)


