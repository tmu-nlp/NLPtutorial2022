import sys
from collections import defaultdict

my_dict = defaultdict(lambda: 0)#デフォルトの値を0にする
my_file = open(sys.argv[1], "r").readlines()#コマンドラインで指定したファイルを読み込む

def word_probability():
    count = 0  
    p_dict = defaultdict(lambda: 0)
    for line in my_file:
        my_list = line.strip().split()#空白区切りで単語を配列に格納

        for value in my_list:
            my_dict[value] += 1
            count += 1 #1単語1カウント

        count += 1#文末記号1カウント

    for key, value in enumerate(my_dict):
                p_dict[value] = '{:.6f}'.format(float(key)/count)
    
    return p_dict

p_dict = word_probability()

for key, value in sorted(p_dict.items()):##キー・値の各組をキー順に表示する
    print(key,value)







