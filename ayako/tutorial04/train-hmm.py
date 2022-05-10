from collections import defaultdict

class train_HMM:
    def __init__(self):
        self.emit = defaultdict(lambda : 0)#生成確率辞書
        self.trasition = defaultdict(lambda : 0)#遷移確率辞書
        self.context = defaultdict(lambda : 0)#文脈数え上げ

    def train_word(self,train_file):#学習
        for line in train_file:
            previous = "<s>"#一個前の単語のタグ
            self.context[previous] += 1#文頭記号をカウント
            wordtags = line.strip().split()#空白区切りで単語_タグのペアを配列に格納
            for wordtag in wordtags:
                word_and_tag = wordtag.split("_")#アンスコ区切りで単語，タグをそれぞれ配列に格納
                self.trasition[previous + " " + word_and_tag[1]] += 1#遷移を数え上げ
                self.context[word_and_tag[1]] += 1#文脈を数え上げ
                self.emit[word_and_tag[1] + " " + word_and_tag[0]] += 1#生成を数え上げ
                previous = word_and_tag[1]#次の単語のために
            self.trasition[previous + " </s>"] += 1#"文末単語 文末記号"は別で追加カウント
        #write_emit("output.txt")

    def write_transiton(self):#遷移確率を出力
        for key, value in self.trasition.items():
            pre_and_word = key.split()
            print("T", key, value/float(self.context[pre_and_word[0]]))

    def write_emit(self):#生成確率を出力
        for key, value in self.emit.items():
            tag_and_word = key.split()
            print("E", key, value/float(self.context[tag_and_word[0]]))

if  __name__ == "__main__":
    path = "../../../nlptutorial/"
    #train_file = open(path + "test/05-train-input.txt", "r").readlines()
    train_file = open(path + "data/wiki-en-train.norm_pos", "r").readlines()
    x = train_HMM()
    x.train_word(train_file)
    x.write_transiton()
    x.write_emit()
