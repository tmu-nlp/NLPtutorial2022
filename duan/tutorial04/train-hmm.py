from collections import defaultdict

emit = defaultdict(lambda: 0) # 生成を格納する
transition = defaultdict(lambda: 0) # 遷移を格納する
context = defaultdict(lambda: 0) # 文脈の頻度を格納する

file = open('./nlptutorial/data/wiki-en-train.norm_pos')
for line in file:
    previous = '<s>' # 文頭記号とおく
    context[previous] += 1
    wordtags = line.rstrip().split(' ') # 改行コードを除去して、リストに分割する
    for wordtag in wordtags:
        word, tag = wordtag.split('_') # 単語と品詞に分ける
        transition[previous + ' ' + tag] += 1 # 遷移を数え上げる
        context[tag] += 1 # 文脈を数え上げる
        emit[tag + ' ' + word] += 1 # 生成を数え上げる
        previous = tag
    transition[previous + ' </s>'] += 1 # 文末記号を追加する

model = open('./NLPtutorial2022/duan/tutorial04/model.txt','w')
# 遷移確率を出力
for key, value in sorted(transition.items()):
    previous, word = key.split(' ')
    print('T ' + key  + ' ' + str(round(value/context[previous],6)), file=model)

# 生成確率を出力
for key, value in sorted(emit.items()):
    tag, word = key.split(' ')
    print('E ' + key +  ' ' + str(round(value/context[tag],6)), file=model)
