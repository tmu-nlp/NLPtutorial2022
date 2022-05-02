import sys
from collections import defaultdict

def make_wordslist(data_file):
    my_list = line.strip().split()#空白区切りで単語を配列に格納
    my_list.append("</s>")
    my_list.insert(0,"<s>") #文頭記号を先頭に追加
    return words_list

def make_unidict(data_file):
    uni_dict = defaultdict(lambda: 0)
    uni_freq = defaultdict(lambda: 0)
    my_dict1 = defaultdict(lambda: 0)
    line_cnt = 0

    for line in data_file:
        my_list = make_wordslist(data_file)
        for value in my_list:
            my_dict1[value] += 1
        line_cnt += 1 #行数カウント
            
    count = sum(my_dict1.values())#総単語数

    for key, value in sorted(my_dict1.items()): #value:頻度の辞書
        uni_freq[key] = value
    
    for key, value in sorted(my_dict1.items()): #value:確率の辞書
        uni_dict[key] = '{:.6f}'.format(float(value)/(count - line_cnt)) #文頭記号の数=line_cntを分母から引く

    return uni_dict, uni_freq

def make_bidict(data_file,uni_freq):
    bi_dict = defaultdict(lambda: 0)
    my_dict2 = defaultdict(lambda: 0)

    for line in data_file:
        words_list = make_wordslist(data_file)
        for i in range(1,len(words_list)):#i-1したときに文頭記号を含まないように
            pair = words_list[i-1] + " " + words_list[i]
            my_dict2[pair] += 1
    
    for key, value in sorted(my_dict2.items()):
        former = key.split()[0] #2語を配列にして1個目の単語を取り出す
        bi_dict[key] = '{:.6f}'.format(float(value)/float(uni_freq[former]))

    return bi_dict

data_file = open(sys.argv[1], "r").readlines()
uni_dict, uni_freq = make_unidict(data_file)
bi_dict = make_bidict(data_file, uni_freq)

for key, value in sorted(bi_dict.items() | uni_dict.items()):
    print(key,value)

