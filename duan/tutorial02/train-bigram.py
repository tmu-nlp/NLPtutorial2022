from collections import defaultdict

#デフォルト値を設定する
counts = defaultdict(lambda:0)
context_counts = defaultdict(lambda:0)
#ファイルの読み込み
training_file = open('./nlptutorial/data/wiki-en-train.word').readlines()

for line in training_file:
    words = line.split() #単語を分ける
    words.append('</s>') #文末記号
    words.insert(0,'<s>') #文頭記号
    for i in range(1,len(words)):
        counts[words[i-1]+' '+words[i]] += 1 #2-gramの分子
        context_counts[words[i-1]] += 1 #2-gramの分母
        counts[words[i]] += 1 #1-gramの分子
        context_counts[''] += 1  #1-gramの分母

#ファイルの書き込み
output = open('./nlptutorial/data/train-bigram.txt','w')

for ngram, count in sorted(counts.items()):
    context = ngram.split(' ') #単語を分ける
    context.pop() #末尾の要素を削除する
    context = ' '.join(context) #文字列を連結する
    probability = float(counts[ngram]/context_counts[context])
    output.write (ngram + '  ' + '{:.6f}'.format(probability) + '\n') #文字列を書き込む
