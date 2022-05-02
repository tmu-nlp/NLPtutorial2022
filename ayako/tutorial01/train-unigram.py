import sys
from collections import defaultdict

my_dict = defaultdict(lambda: 0)#デフォルトの値を0にする
my_file = open(sys.argv[1], "r").readlines()#コマンドラインで指定したファイルを読み込む
count = 0  

for line in my_file:
    my_list = line.strip().split()#空白区切りで単語を配列に格納
    my_list.append("</s>")
    for value in my_list:
        my_dict[value] += 1
        count += 1 #1単語1カウント

for key, value in sorted(my_dict.items()):##キー・値の各組をキー順に表示する
    print(key,'{:.6f}'.format(float(value)/count))







